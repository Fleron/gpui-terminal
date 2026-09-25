//! Main terminal view component for GPUI.
//!
//! This module provides [`TerminalView`], the primary component for embedding terminals
//! in GPUI applications. It manages:
//!
//! - **I/O Streams**: Accepts arbitrary [`Read`]/[`Write`]
//!   streams, allowing integration with any PTY implementation
//! - **Event Handling**: Keyboard and mouse input, with configurable callbacks
//! - **Rendering**: Efficient canvas-based rendering via [`TerminalRenderer`]
//! - **Configuration**: Font, colors, dimensions, and padding via [`TerminalConfig`]
//!
//! # Architecture
//!
//! The terminal uses a push-based async I/O architecture:
//!
//! 1. A background thread reads bytes from the PTY stdout in 4KB chunks
//! 2. Bytes are sent through a [flume](https://docs.rs/flume) channel to an async task
//! 3. The async task processes bytes through the VTE parser and calls `cx.notify()`
//! 4. GPUI repaints the terminal with the updated grid
//!
//! This approach ensures the terminal only wakes when data arrives, avoiding polling.
//!
//! # Thread Safety
//!
//! - [`TerminalView`] itself is not `Send` (it contains GPUI handles)
//! - The stdin writer is wrapped in `Arc<parking_lot::Mutex<>>` for thread-safe writes
//! - Callbacks ([`ResizeCallback`], [`KeyHandler`]) must be `Send + Sync`
//!
//! # Example
//!
//! ```ignore
//! use gpui::{Context, Edges, px};
//! use gpui_terminal::{ColorPalette, TerminalConfig, TerminalView};
//!
//! // In a GPUI window context:
//! let terminal = cx.new(|cx| {
//!     TerminalView::new(pty_writer, pty_reader, TerminalConfig::default(), cx)
//!         .with_resize_callback(move |cols, rows| {
//!             // Notify PTY of new dimensions
//!         })
//!         .with_exit_callback(|_, cx| {
//!             cx.quit();
//!         })
//! });
//!
//! // Focus the terminal to receive keyboard input
//! terminal.read(cx).focus_handle().focus(window);
//! ```

use crate::colors::ColorPalette;
use crate::event::{GpuiEventProxy, TerminalEvent};
use crate::input::keystroke_to_bytes;
use crate::mouse::{
    ScrollAction, encode_modifiers, mouse_button_report, mouse_motion_report, pixel_to_cell,
    scroll_action,
};
use crate::render::TerminalRenderer;
use crate::terminal::TerminalState;
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::index::{Column, Line, Point as AlacPoint, Side};
use alacritty_terminal::selection::{Selection, SelectionType};
use alacritty_terminal::term::TermMode;
use gpui::{
    AsyncApp, Bounds, Context, Edges, FocusHandle, InteractiveElement, IntoElement, KeyDownEvent,
    Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement, Pixels,
    Point, Render, ScrollWheelEvent, Styled, Task, WeakEntity, Window, canvas, div, px, rgb,
};
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

/// Configuration for terminal creation and runtime updates.
///
/// This struct defines the terminal's appearance and behavior, including
/// grid dimensions, font settings, scrollback buffer, and color scheme.
///
/// # Default Values
///
/// | Field | Default |
/// |-------|---------|
/// | `cols` | 80 |
/// | `rows` | 24 |
/// | `font_family` | "monospace" |
/// | `font_size` | 14px |
/// | `scrollback` | 10000 |
/// | `line_height_multiplier` | 1.0 |
/// | `padding` | 0px all sides |
/// | `colors` | Default palette |
///
/// # Example
///
/// ```ignore
/// use gpui::{Edges, px};
/// use gpui_terminal::{ColorPalette, TerminalConfig};
///
/// let config = TerminalConfig {
///     cols: 120,
///     rows: 40,
///     font_family: "JetBrains Mono".into(),
///     font_size: px(13.0),
///     scrollback: 50000,
///     line_height_multiplier: 1.0,
///     padding: Edges::all(px(10.0)),
///     colors: ColorPalette::builder()
///         .background(0x1a, 0x1a, 0x1a)
///         .foreground(0xe0, 0xe0, 0xe0)
///         .build(),
/// };
/// ```
///
/// # Runtime Updates
///
/// Configuration can be updated at runtime via [`TerminalView::update_config`].
/// This is useful for implementing features like dynamic font sizing:
///
/// ```ignore
/// terminal.update(cx, |terminal, cx| {
///     let mut config = terminal.config().clone();
///     config.font_size += px(1.0);
///     terminal.update_config(config, cx);
/// });
/// ```
#[derive(Clone, Debug)]
pub struct TerminalConfig {
    /// Number of columns (character width) in the terminal
    pub cols: usize,

    /// Number of rows (lines) in the terminal
    pub rows: usize,

    /// Font family name (e.g., "Fira Code", "JetBrains Mono")
    pub font_family: String,

    /// Font size in pixels
    pub font_size: Pixels,

    /// Maximum number of scrollback lines to keep in history
    pub scrollback: usize,

    /// Multiplier for line height to accommodate tall glyphs (e.g., nerd fonts)
    /// Default is 1.0 (no extra height)
    pub line_height_multiplier: f32,

    /// Padding around the terminal content (top, right, bottom, left)
    /// The padding area renders with the terminal's background color
    pub padding: Edges<Pixels>,

    /// Color palette for terminal colors (16 ANSI colors, 256 extended colors,
    /// foreground, background, and cursor colors)
    pub colors: ColorPalette,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            cols: 80,
            rows: 24,
            font_family: "monospace".into(),
            font_size: px(14.0),
            scrollback: 10000,
            line_height_multiplier: 1.0,
            padding: Edges::all(px(0.0)),
            colors: ColorPalette::default(),
        }
    }
}

/// Callback type for PTY resize notifications.
///
/// This callback is invoked when the terminal grid dimensions change,
/// typically due to window resizing. The callback receives the new
/// column and row counts.
///
/// # Arguments
///
/// * `cols` - New number of columns (characters wide)
/// * `rows` - New number of rows (lines tall)
///
/// # Thread Safety
///
/// This callback must be `Send + Sync` as it may be called from the render thread.
///
/// # Example
///
/// ```ignore
/// use portable_pty::PtySize;
///
/// let pty = Arc::new(Mutex::new(pty_master));
/// let pty_clone = pty.clone();
///
/// terminal.with_resize_callback(move |cols, rows| {
///     pty_clone.lock().resize(PtySize {
///         cols: cols as u16,
///         rows: rows as u16,
///         pixel_width: 0,
///         pixel_height: 0,
///     }).ok();
/// });
/// ```
pub type ResizeCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Callback type for key event interception.
///
/// This callback is invoked before the terminal processes a key event,
/// allowing you to intercept and handle specific key combinations.
///
/// # Arguments
///
/// * `event` - The key down event from GPUI
///
/// # Returns
///
/// * `true` - Consume the event (terminal will not process it)
/// * `false` - Let the terminal handle the event normally
///
/// # Thread Safety
///
/// This callback must be `Send + Sync`.
///
/// # Example
///
/// ```ignore
/// terminal.with_key_handler(|event| {
///     let keystroke = &event.keystroke;
///
///     // Intercept Ctrl++ for font size increase
///     if keystroke.modifiers.control && (keystroke.key == "+" || keystroke.key == "=") {
///         // Handle font size increase
///         return true; // Consume the event
///     }
///
///     // Intercept Ctrl+- for font size decrease
///     if keystroke.modifiers.control && keystroke.key == "-" {
///         // Handle font size decrease
///         return true;
///     }
///
///     false // Let terminal handle all other keys
/// });
/// ```
pub type KeyHandler = Box<dyn Fn(&KeyDownEvent) -> bool + Send + Sync>;

/// Callback for terminal bell events.
///
/// This callback is invoked when the terminal bell is triggered (BEL character,
/// ASCII 0x07), allowing you to play a sound or show a visual indicator.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
///
/// # Example
///
/// ```ignore
/// terminal.with_bell_callback(|window, cx| {
///     // Option 1: Visual bell (flash the window or show an indicator)
///     // Option 2: Play a sound
///     // Option 3: Notify the user via system notification
/// });
/// ```
pub type BellCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>)>;

/// Callback for terminal title changes.
///
/// This callback is invoked when the terminal title changes via escape sequences
/// (OSC 0, OSC 2), allowing you to update the window or tab title.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
/// * `title` - The new title string
///
/// # Example
///
/// ```ignore
/// terminal.with_title_callback(|window, cx, title| {
///     // Update the window title
///     // Or update a tab label in a tabbed interface
///     println!("Terminal title changed to: {}", title);
/// });
/// ```
pub type TitleCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>, &str)>;

/// Callback for clipboard store requests.
///
/// This callback is invoked when the terminal wants to store data to the clipboard
/// via OSC 52 escape sequence. Applications like tmux and vim can use this to
/// copy text to the system clipboard.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
/// * `text` - The text to store in the clipboard
///
/// # Example
///
/// ```ignore
/// use gpui_terminal::Clipboard;
///
/// terminal.with_clipboard_store_callback(|window, cx, text| {
///     if let Ok(mut clipboard) = Clipboard::new() {
///         clipboard.copy(text).ok();
///     }
/// });
/// ```
pub type ClipboardStoreCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>, &str)>;

/// Callback for terminal exit events.
///
/// This callback is invoked when the terminal process exits (e.g., shell exits,
/// process terminates). This is detected when the PTY reader reaches EOF.
///
/// # Arguments
///
/// * `window` - The GPUI window
/// * `cx` - The context for the TerminalView
///
/// # Example
///
/// ```ignore
/// terminal.with_exit_callback(|window, cx| {
///     // Option 1: Quit the application
///     cx.quit();
///
///     // Option 2: Close this terminal tab/pane
///     // terminal_manager.close_terminal(terminal_id);
///
///     // Option 3: Show an exit message
///     // show_notification("Terminal exited");
/// });
/// ```
pub type ExitCallback = Box<dyn Fn(&mut Window, &mut Context<TerminalView>)>;

/// The main terminal view component for GPUI applications.
///
/// `TerminalView` is a GPUI entity that implements the [`Render`] trait,
/// providing a complete terminal emulator that can be embedded in any GPUI application.
///
/// # Responsibilities
///
/// - **Terminal State**: Manages the grid, cursor, and colors via [`TerminalState`]
/// - **I/O Streams**: Reads from PTY stdout and writes to PTY stdin
/// - **Event Handling**: Processes keyboard, mouse, and resize events
/// - **Rendering**: Paints text, backgrounds, and cursor via [`TerminalRenderer`]
/// - **Callbacks**: Dispatches events to user-provided callbacks
///
/// # Creating a Terminal
///
/// Use [`TerminalView::new`] within a GPUI entity context:
///
/// ```ignore
/// let terminal = cx.new(|cx| {
///     TerminalView::new(writer, reader, config, cx)
///         .with_resize_callback(resize_callback)
///         .with_exit_callback(|_, cx| cx.quit())
/// });
/// ```
///
/// # Focus
///
/// The terminal must be focused to receive keyboard input:
///
/// ```ignore
/// terminal.read(cx).focus_handle().focus(window);
/// ```
///
/// # Callbacks
///
/// Configure behavior through builder methods:
///
/// - [`with_resize_callback`](Self::with_resize_callback) - PTY size changes
/// - [`with_exit_callback`](Self::with_exit_callback) - Process exit
/// - [`with_key_handler`](Self::with_key_handler) - Key event interception
/// - [`with_bell_callback`](Self::with_bell_callback) - Terminal bell
/// - [`with_title_callback`](Self::with_title_callback) - Title changes
/// - [`with_clipboard_store_callback`](Self::with_clipboard_store_callback) - Clipboard writes
///
/// # Thread Safety
///
/// `TerminalView` is not `Send` as it contains GPUI handles. The stdin writer
/// is internally wrapped in `Arc<parking_lot::Mutex<>>` for safe concurrent access.
pub struct TerminalView {
    /// The terminal state managing the grid and VTE parser
    state: TerminalState,

    /// The renderer for drawing terminal content
    renderer: TerminalRenderer,

    /// Geometry measured by the canvas, in window coordinates.
    geometry: Arc<parking_lot::Mutex<Option<TerminalGeometry>>>,

    /// Unconsumed wheel movement in pixels, retained across trackpad events.
    scroll_accum: f32,

    /// Window position of the current left-button press.
    drag_origin: Option<Point<Pixels>>,

    /// Whether a drag or multiple click has started a selection.
    selecting: bool,

    /// Whether the current left-button press was sent to a mouse-aware application.
    reporting_mouse_down: bool,

    /// Last grid cell sent in a mouse motion report.
    last_reported_cell: Option<AlacPoint>,

    /// Focus handle for keyboard event handling
    focus_handle: FocusHandle,

    /// Writer for sending input to the terminal process
    stdin_writer: Arc<parking_lot::Mutex<Box<dyn Write + Send>>>,

    /// Receiver for terminal events from the event proxy
    event_rx: mpsc::Receiver<TerminalEvent>,

    /// Configuration used to create this terminal
    config: TerminalConfig,

    /// Async task that reads bytes and notifies the view (push-based)
    #[allow(dead_code)]
    _reader_task: Task<()>,

    /// Callback to notify the PTY about size changes
    resize_callback: Option<Arc<ResizeCallback>>,

    /// Optional callback to intercept key events before terminal processing
    key_handler: Option<Arc<KeyHandler>>,

    /// Callback for terminal bell events
    bell_callback: Option<BellCallback>,

    /// Callback for terminal title changes
    title_callback: Option<TitleCallback>,

    /// Callback for clipboard store requests
    clipboard_store_callback: Option<ClipboardStoreCallback>,

    /// Callback for terminal exit events
    exit_callback: Option<ExitCallback>,
}

#[derive(Clone, Copy)]
struct TerminalGeometry {
    origin: Point<Pixels>,
    cell_width: Pixels,
    cell_height: Pixels,
}

/// Hit test a window position against the visible grid, including scrollback.
fn hit_cell(
    position: Point<Pixels>,
    geometry: TerminalGeometry,
    cols: usize,
    rows: usize,
    display_offset: usize,
) -> (AlacPoint, Side) {
    let x = (position.x - geometry.origin.x) / geometry.cell_width;
    let y = (position.y - geometry.origin.y) / geometry.cell_height;
    let last_col = cols.saturating_sub(1);
    let last_row = rows.saturating_sub(1);

    if y >= rows as f32 {
        return (
            AlacPoint::new(
                Line(last_row as i32 - display_offset as i32),
                Column(last_col),
            ),
            Side::Right,
        );
    }

    let col = (x.floor().max(0.0) as usize).min(last_col);
    let row = (y.floor().max(0.0) as usize).min(last_row);
    let side = if y < 0.0 || x < 0.0 {
        Side::Left
    } else if x >= cols as f32 || x.fract() >= 0.5 {
        Side::Right
    } else {
        Side::Left
    };

    (
        AlacPoint::new(Line(row as i32 - display_offset as i32), Column(col)),
        side,
    )
}

impl TerminalView {
    /// Create a new terminal with provided I/O streams.
    ///
    /// This method initializes a new terminal emulator with the given stdin writer
    /// and stdout reader. It spawns a background task to read from stdout and
    /// process incoming bytes through the VTE parser.
    ///
    /// # Arguments
    ///
    /// * `stdin_writer` - Writer for sending input bytes to the terminal process
    /// * `stdout_reader` - Reader for receiving output bytes from the terminal process
    /// * `config` - Terminal configuration (dimensions, font, etc.)
    /// * `cx` - GPUI context for this view
    ///
    /// # Returns
    ///
    /// A new `TerminalView` instance ready to be rendered.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // In a GPUI window context:
    /// let terminal = cx.new(|cx| {
    ///     TerminalView::new(stdin_writer, stdout_reader, TerminalConfig::default(), cx)
    /// });
    /// ```
    pub fn new<W, R>(
        stdin_writer: W,
        stdout_reader: R,
        config: TerminalConfig,
        cx: &mut Context<Self>,
    ) -> Self
    where
        W: Write + Send + 'static,
        R: Read + Send + 'static,
    {
        // Create event channel for terminal events
        let (event_tx, event_rx) = mpsc::channel();

        // Clone event_tx for the reader task to send Exit event when PTY closes
        let exit_event_tx = event_tx.clone();

        // Create event proxy for alacritty
        let event_proxy = GpuiEventProxy::new(event_tx);

        // Create terminal state
        let state = TerminalState::new_with_scrollback(
            config.cols,
            config.rows,
            config.scrollback,
            event_proxy,
        );

        // Create renderer with font settings and color palette
        let renderer = TerminalRenderer::new(
            config.font_family.clone(),
            config.font_size,
            config.line_height_multiplier,
            config.colors.clone(),
        );

        // Create focus handle
        let focus_handle = cx.focus_handle();

        // Wrap stdin writer in Arc<Mutex> for thread-safe access
        let stdin_writer = Arc::new(parking_lot::Mutex::new(
            Box::new(stdin_writer) as Box<dyn Write + Send>
        ));

        // Create async channel for bytes (push-based notification)
        // Using flume instead of smol::channel because flume is executor-agnostic
        // and properly wakes GPUI's async executor when data arrives
        let (bytes_tx, bytes_rx) = flume::unbounded::<Vec<u8>>();

        // Spawn background thread to read from stdout
        // This thread sends bytes through the async channel
        thread::spawn(move || {
            Self::read_stdout_blocking(stdout_reader, bytes_tx);
        });

        // Spawn async task that awaits on the channel and notifies the view
        // This is push-based: the task blocks until bytes arrive, then immediately notifies
        let reader_task = cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            loop {
                // Wait for bytes from the background reader (blocks until data arrives)
                match bytes_rx.recv_async().await {
                    Ok(bytes) => {
                        // Process bytes and notify the view
                        let result = this.update(cx, |view: &mut Self, cx: &mut Context<Self>| {
                            view.state.process_bytes(&bytes);
                            cx.notify();
                        });
                        if result.is_err() {
                            // View was dropped, exit
                            break;
                        }
                    }
                    Err(_) => {
                        // Channel closed - PTY has finished, send Exit event
                        let _ = exit_event_tx.send(TerminalEvent::Exit);
                        // Notify view to process the Exit event
                        let _ = this.update(cx, |_view, cx: &mut Context<Self>| {
                            cx.notify();
                        });
                        break;
                    }
                }
            }
        });

        Self {
            state,
            renderer,
            geometry: Arc::new(parking_lot::Mutex::new(None)),
            scroll_accum: 0.0,
            drag_origin: None,
            selecting: false,
            reporting_mouse_down: false,
            last_reported_cell: None,
            focus_handle,
            stdin_writer,
            event_rx,
            config,
            _reader_task: reader_task,
            resize_callback: None,
            key_handler: None,
            bell_callback: None,
            title_callback: None,
            clipboard_store_callback: None,
            exit_callback: None,
        }
    }

    /// Set a callback to be invoked when the terminal is resized.
    ///
    /// This callback should resize the underlying PTY to match the new dimensions.
    /// The callback receives (cols, rows) as arguments.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called with (cols, rows) on resize
    pub fn with_resize_callback(
        mut self,
        callback: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> Self {
        self.resize_callback = Some(Arc::new(Box::new(callback)));
        self
    }

    /// Set a callback to intercept key events before terminal processing.
    ///
    /// The callback receives the key event and should return `true` to consume
    /// the event (prevent the terminal from processing it), or `false` to allow
    /// normal terminal processing.
    ///
    /// # Arguments
    ///
    /// * `handler` - A function that receives key events and returns whether to consume them
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_key_handler(|event| {
    ///     // Handle Ctrl++ to increase font size
    ///     if event.keystroke.modifiers.control && event.keystroke.key == "+" {
    ///         // Handle the event
    ///         return true; // Consume the event
    ///     }
    ///     false // Let terminal handle it
    /// })
    /// ```
    pub fn with_key_handler(
        mut self,
        handler: impl Fn(&KeyDownEvent) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.key_handler = Some(Arc::new(Box::new(handler)));
        self
    }

    /// Set a callback to be invoked when the terminal bell is triggered.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// allowing you to play a sound or show a visual indicator.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called when the bell is triggered
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_bell_callback(|window, cx| {
    ///     // Play a sound or flash the screen
    /// })
    /// ```
    pub fn with_bell_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>) + 'static,
    ) -> Self {
        self.bell_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal title changes.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// along with the new title string.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called with the new title
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_title_callback(|window, cx, title| {
    ///     // Update window title or tab title
    /// })
    /// ```
    pub fn with_title_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>, &str) + 'static,
    ) -> Self {
        self.title_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal wants to store data to the clipboard.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// along with the text to store. This is typically triggered by OSC 52 escape sequences.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called with the text to store
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_clipboard_store_callback(|window, cx, text| {
    ///     // Store text to system clipboard
    /// })
    /// ```
    pub fn with_clipboard_store_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>, &str) + 'static,
    ) -> Self {
        self.clipboard_store_callback = Some(Box::new(callback));
        self
    }

    /// Set a callback to be invoked when the terminal process exits.
    ///
    /// The callback receives a mutable reference to the window and context,
    /// allowing you to close the terminal view or show an exit message.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function that will be called when the process exits
    ///
    /// # Example
    ///
    /// ```ignore
    /// terminal.with_exit_callback(|window, cx| {
    ///     // Close the terminal tab or show exit message
    /// })
    /// ```
    pub fn with_exit_callback(
        mut self,
        callback: impl Fn(&mut Window, &mut Context<TerminalView>) + 'static,
    ) -> Self {
        self.exit_callback = Some(Box::new(callback));
        self
    }

    /// Background thread that reads from stdout.
    ///
    /// This function runs in a background thread, continuously reading bytes
    /// from the stdout reader and sending them through the async channel.
    /// The async channel allows the main async task to be woken up immediately
    /// when data arrives (push-based).
    fn read_stdout_blocking<R: Read + Send + 'static>(
        mut stdout_reader: R,
        bytes_tx: flume::Sender<Vec<u8>>,
    ) {
        let mut buffer = [0u8; 4096];

        loop {
            match stdout_reader.read(&mut buffer) {
                Ok(0) => {
                    // EOF - channel will be dropped, signaling completion
                    break;
                }
                Ok(n) => {
                    // Send bytes to the async task
                    let bytes = buffer[..n].to_vec();
                    if bytes_tx.send(bytes).is_err() {
                        break; // Channel closed
                    }
                }
                Err(_) => {
                    // Read error
                    break;
                }
            }
        }
    }

    /// Handle keyboard input events.
    ///
    /// Converts GPUI keystrokes to terminal escape sequences and writes them
    /// to the stdin writer. If a key handler is set and returns true, the event
    /// is consumed and not sent to the terminal.
    fn on_key_down(&mut self, event: &KeyDownEvent, _window: &mut Window, cx: &mut Context<Self>) {
        // Check if key handler wants to consume this event
        if let Some(ref handler) = self.key_handler
            && handler(event)
        {
            if !event.keystroke.modifiers.platform {
                self.scroll_to_bottom_and_clear_selection(cx);
            }
            return; // Event consumed by handler
        }

        if let Some(bytes) = keystroke_to_bytes(&event.keystroke, self.state.mode()) {
            self.scroll_to_bottom_and_clear_selection(cx);

            let mut writer = self.stdin_writer.lock();
            let _ = writer.write_all(&bytes);
            let _ = writer.flush();
        }
    }

    /// Focus the view, report mouse input to applications, or begin a selection.
    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.focus(&self.focus_handle);
        self.last_reported_cell = None;
        let Some(geometry) = *self.geometry.lock() else {
            return;
        };
        let mode = self.state.mode();
        let (cols, rows, offset) = self.state.with_term(|term| {
            (
                term.columns(),
                term.screen_lines(),
                term.grid().display_offset(),
            )
        });

        if mode.intersects(TermMode::MOUSE_MODE) && !event.modifiers.shift {
            self.drag_origin = None;
            self.selecting = false;
            self.reporting_mouse_down = true;
            let (point, _) = hit_cell(event.position, geometry, cols, rows, 0);
            self.send_mouse_button(true, point, event.modifiers, mode);
            cx.notify();
            return;
        }

        self.reporting_mouse_down = false;
        let (point, side) = hit_cell(event.position, geometry, cols, rows, offset);
        self.drag_origin = Some(event.position);
        self.selecting = false;
        self.state.with_term_mut(|term| {
            if event.click_count >= 2 {
                let ty = if event.click_count == 2 {
                    SelectionType::Semantic
                } else {
                    SelectionType::Lines
                };
                term.selection = Some(Selection::new(ty, point, side));
                self.selecting = true;
            } else if event.modifiers.shift
                && let Some(selection) = term.selection.as_mut()
            {
                selection.update(point, side);
                self.selecting = true;
            } else {
                term.selection = None;
            }
        });
        cx.notify();
    }

    /// Finalize a selection or report a mouse release.
    fn on_mouse_up(&mut self, event: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.last_reported_cell = None;
        let was_selecting = self.selecting;
        let had_drag_origin = self.drag_origin.take().is_some();
        let was_reporting = std::mem::replace(&mut self.reporting_mouse_down, false);
        self.selecting = false;

        if was_reporting {
            let mode = self.state.mode();
            if mode.intersects(TermMode::MOUSE_MODE)
                && let Some(geometry) = *self.geometry.lock()
            {
                let (cols, rows) = self
                    .state
                    .with_term(|term| (term.columns(), term.screen_lines()));
                let (point, _) = hit_cell(event.position, geometry, cols, rows, 0);
                self.send_mouse_button(false, point, event.modifiers, mode);
            }
            cx.notify();
            return;
        }
        if !had_drag_origin {
            return;
        }

        if was_selecting && let Some(geometry) = *self.geometry.lock() {
            self.state.with_term_mut(|term| {
                let (point, side) = hit_cell(
                    event.position,
                    geometry,
                    term.columns(),
                    term.screen_lines(),
                    term.grid().display_offset(),
                );
                if let Some(selection) = term.selection.as_mut() {
                    selection.update(point, side);
                }
            });
        }
        self.state.with_term_mut(|term| {
            if term.selection.as_ref().is_some_and(Selection::is_empty) {
                term.selection = None;
            }
        });
        cx.notify();
    }

    /// Update the selection once the left-button drag passes two pixels.
    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.reporting_mouse_down {
            if event.dragging()
                && self
                    .state
                    .mode()
                    .intersects(TermMode::MOUSE_DRAG | TermMode::MOUSE_MOTION)
            {
                self.send_mouse_motion(Some(MouseButton::Left), event);
            }
            return;
        }
        if self.drag_origin.is_none()
            && !event.dragging()
            && !event.modifiers.shift
            && self.state.mode().contains(TermMode::MOUSE_MOTION)
        {
            self.send_mouse_motion(None, event);
            return;
        }

        let Some(origin) = self.drag_origin else {
            return;
        };
        if !event.dragging() {
            return;
        }
        let Some(geometry) = *self.geometry.lock() else {
            return;
        };
        let allow_scroll = !self.state.mode().contains(TermMode::ALT_SCREEN);

        if !self.selecting {
            let dx: f32 = (event.position.x - origin.x).into();
            let dy: f32 = (event.position.y - origin.y).into();
            if dx * dx + dy * dy < 4.0 {
                return;
            }
        }

        self.state.with_term_mut(|term| {
            let cols = term.columns();
            let rows = term.screen_lines();
            if !self.selecting {
                let offset = term.grid().display_offset();
                let (point, side) = hit_cell(origin, geometry, cols, rows, offset);
                let ty = if event.modifiers.alt {
                    SelectionType::Block
                } else {
                    SelectionType::Simple
                };
                term.selection = Some(Selection::new(ty, point, side));
                self.selecting = true;
            }
            if allow_scroll {
                let grid_top = geometry.origin.y;
                let grid_bottom = grid_top + geometry.cell_height * rows as f32;
                let overshoot = if event.position.y < grid_top {
                    (grid_top - event.position.y) / geometry.cell_height
                } else if event.position.y >= grid_bottom {
                    (grid_bottom - event.position.y) / geometry.cell_height
                } else {
                    0.0
                };
                if overshoot != 0.0 {
                    let lines = (overshoot.abs().round() as i32).clamp(1, 3);
                    term.scroll_display(Scroll::Delta(lines * overshoot.signum() as i32));
                }
            }
            let offset = term.grid().display_offset();
            let (point, side) = hit_cell(event.position, geometry, cols, rows, offset);
            if let Some(selection) = term.selection.as_mut() {
                selection.update(point, side);
            }
        });
        cx.notify();
    }

    fn send_mouse_button(
        &self,
        pressed: bool,
        point: AlacPoint,
        modifiers: Modifiers,
        mode: TermMode,
    ) {
        let modifiers = encode_modifiers(modifiers.shift, modifiers.alt, modifiers.control);
        if let Some(bytes) = mouse_button_report(MouseButton::Left, pressed, point, modifiers, mode)
        {
            let mut writer = self.stdin_writer.lock();
            let _ = writer.write_all(&bytes);
            let _ = writer.flush();
        }
    }

    fn send_mouse_motion(&mut self, pressed_button: Option<MouseButton>, event: &MouseMoveEvent) {
        let Some(geometry) = *self.geometry.lock() else {
            return;
        };
        let mode = self.state.mode();
        let (cols, rows) = self
            .state
            .with_term(|term| (term.columns(), term.screen_lines()));
        let (point, _) = hit_cell(event.position, geometry, cols, rows, 0);
        let modifiers = encode_modifiers(
            event.modifiers.shift,
            event.modifiers.alt,
            event.modifiers.control,
        );
        if let Some(bytes) = mouse_motion_report(
            pressed_button,
            point,
            modifiers,
            mode,
            &mut self.last_reported_cell,
        ) {
            let mut writer = self.stdin_writer.lock();
            let _ = writer.write_all(&bytes);
            let _ = writer.flush();
        }
    }

    /// Handle scroll events.
    ///
    fn on_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(geometry) = *self.geometry.lock() else {
            return;
        };
        let cell_height: f32 = geometry.cell_height.into();
        if cell_height <= 0.0 {
            return;
        }

        // GPUI's positive Y delta moves content down, exposing older lines.
        let delta: f32 = event.delta.pixel_delta(geometry.cell_height).y.into();
        self.scroll_accum += delta;
        let lines = (self.scroll_accum / cell_height).trunc() as i32;
        if lines == 0 {
            return;
        }
        self.scroll_accum -= lines as f32 * cell_height;

        let point = pixel_to_cell(
            event.position,
            geometry.origin,
            geometry.cell_width,
            geometry.cell_height,
        );
        let (cols, rows) = self
            .state
            .with_term(|term| (term.columns(), term.screen_lines()));
        let point = AlacPoint::new(
            Line(point.line.0.min(rows.saturating_sub(1) as i32)),
            Column(point.column.0.min(cols.saturating_sub(1))),
        );

        match scroll_action(lines, point, event.modifiers.shift, self.state.mode()) {
            ScrollAction::Report(bytes) => {
                let mut writer = self.stdin_writer.lock();
                let _ = writer.write_all(&bytes);
                let _ = writer.flush();
            }
            ScrollAction::Local(lines) => {
                self.state
                    .with_term_mut(|term| term.scroll_display(Scroll::Delta(lines)));
                cx.notify();
            }
        }
    }

    /// Process pending terminal events.
    ///
    /// This method drains all available events from the event receiver
    /// and handles them appropriately. Note: bytes are processed in the
    /// async reader task, not here.
    fn process_events(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Process terminal events (from alacritty event proxy)
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                TerminalEvent::Wakeup => {
                    // Terminal has new content - already handled by async task
                }
                TerminalEvent::Bell => {
                    if let Some(ref callback) = self.bell_callback {
                        callback(window, cx);
                    }
                }
                TerminalEvent::Title(title) => {
                    if let Some(ref callback) = self.title_callback {
                        callback(window, cx, &title);
                    }
                }
                TerminalEvent::ClipboardStore(text) => {
                    if let Some(ref callback) = self.clipboard_store_callback {
                        callback(window, cx, &text);
                    }
                }
                TerminalEvent::ClipboardLoad => {
                    // Terminal wants to load data from clipboard
                    // TODO: Implement clipboard integration
                }
                TerminalEvent::Exit => {
                    if let Some(ref callback) = self.exit_callback {
                        callback(window, cx);
                    }
                }
            }
        }
    }

    /// Get the current terminal dimensions.
    ///
    /// # Returns
    ///
    /// A tuple of (columns, rows).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.state.cols(), self.state.rows())
    }

    /// Return the selected terminal text, if the selection contains text.
    pub fn selection_text(&self) -> Option<String> {
        self.state
            .with_term(|term| term.selection_to_string().filter(|text| !text.is_empty()))
    }

    /// Whether there is text available to copy from the selection.
    pub fn has_selection(&self) -> bool {
        self.selection_text().is_some()
    }

    /// Clear the selection and repaint if needed.
    pub fn clear_selection(&mut self, cx: &mut Context<Self>) {
        if self
            .state
            .with_term_mut(|term| term.selection.take().is_some())
        {
            cx.notify();
        }
    }

    /// Return to live output and clear the selection before sending input.
    pub fn scroll_to_bottom_and_clear_selection(&mut self, cx: &mut Context<Self>) {
        self.scroll_accum = 0.0;
        let changed = self.state.with_term_mut(|term| {
            let was_scrolled = term.grid().display_offset() > 0;
            if was_scrolled {
                term.scroll_display(Scroll::Bottom);
            }
            let had_selection = term.selection.take().is_some();
            was_scrolled || had_selection
        });
        if changed {
            cx.notify();
        }
    }

    /// Resize the terminal to new dimensions.
    ///
    /// This method should be called when the terminal view size changes.
    /// It updates the internal grid and notifies the terminal process of the new size.
    ///
    /// # Arguments
    ///
    /// * `cols` - New number of columns
    /// * `rows` - New number of rows
    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.state.resize(cols, rows);
    }

    /// Get the current terminal configuration.
    ///
    /// # Returns
    ///
    /// A reference to the current configuration.
    pub fn config(&self) -> &TerminalConfig {
        &self.config
    }

    /// Get the focus handle for this terminal view.
    ///
    /// # Returns
    ///
    /// A reference to the focus handle.
    pub fn focus_handle(&self) -> &FocusHandle {
        &self.focus_handle
    }

    /// Update the terminal configuration.
    ///
    /// This method updates the terminal's configuration, including font settings,
    /// padding, and color palette. Changes take effect on the next render.
    ///
    /// # Arguments
    ///
    /// * `config` - The new configuration to apply
    /// * `cx` - The context for triggering a repaint
    pub fn update_config(&mut self, config: TerminalConfig, cx: &mut Context<Self>) {
        // Update renderer with new font settings and palette
        self.renderer.font_family = config.font_family.clone();
        self.renderer.font_size = config.font_size;
        self.renderer.line_height_multiplier = config.line_height_multiplier;
        self.renderer.palette = config.colors.clone();

        // Store the new config
        self.config = config;

        // Trigger a repaint - cell dimensions will be recalculated via measure_cell()
        cx.notify();
    }

    /// Calculate terminal dimensions from pixel bounds and cell size.
    ///
    /// Helper method to determine how many columns and rows fit in the given bounds.
    #[allow(dead_code)]
    fn calculate_dimensions(&self, bounds: Bounds<Pixels>) -> (usize, usize) {
        let width_f32: f32 = bounds.size.width.into();
        let height_f32: f32 = bounds.size.height.into();
        let cell_width_f32: f32 = self.renderer.cell_width.into();
        let cell_height_f32: f32 = self.renderer.cell_height.into();

        let cols = ((width_f32 / cell_width_f32) as usize).max(1);
        let rows = ((height_f32 / cell_height_f32) as usize).max(1);
        (cols, rows)
    }
}

impl Render for TerminalView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Process any pending events
        self.process_events(window, cx);

        // Get terminal state and renderer for rendering
        let state_arc = self.state.term_arc();
        let renderer = self.renderer.clone();
        let geometry = self.geometry.clone();
        let resize_callback = self.resize_callback.clone();
        let padding = self.config.padding;
        let entity = cx.entity();
        let tracking_pointer = self.drag_origin.is_some() || self.reporting_mouse_down;

        div()
            .size_full()
            .bg(rgb(0x1e1e1e))
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(Self::on_key_down))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_scroll_wheel(cx.listener(Self::on_scroll))
            .child(
                canvas(
                    move |bounds, _window, _cx| bounds,
                    move |bounds, _, window, cx| {
                        use alacritty_terminal::grid::Dimensions;

                        if tracking_pointer {
                            let move_entity = entity.clone();
                            window.on_mouse_event(
                                move |event: &MouseMoveEvent, phase, window, cx| {
                                    if phase == gpui::DispatchPhase::Bubble
                                        && event.dragging()
                                        && !bounds.contains(&event.position)
                                    {
                                        move_entity.update(cx, |view, cx| {
                                            view.on_mouse_move(event, window, cx)
                                        });
                                    }
                                },
                            );
                            window.on_mouse_event(
                                move |event: &MouseUpEvent, phase, window, cx| {
                                    if phase == gpui::DispatchPhase::Bubble
                                        && event.button == MouseButton::Left
                                        && !bounds.contains(&event.position)
                                    {
                                        entity.update(cx, |view, cx| {
                                            view.on_mouse_up(event, window, cx)
                                        });
                                    }
                                },
                            );
                        }

                        // Measure actual cell dimensions from the font
                        let mut measured_renderer = renderer.clone();
                        measured_renderer.measure_cell(window);

                        *geometry.lock() = Some(TerminalGeometry {
                            origin: Point {
                                x: bounds.origin.x + padding.left,
                                y: bounds.origin.y + padding.top,
                            },
                            cell_width: measured_renderer.cell_width,
                            cell_height: measured_renderer.cell_height,
                        });

                        // Calculate available space after padding
                        let available_width: f32 =
                            (bounds.size.width - padding.left - padding.right).into();
                        let available_height: f32 =
                            (bounds.size.height - padding.top - padding.bottom).into();
                        let cell_width_f32: f32 = measured_renderer.cell_width.into();
                        let cell_height_f32: f32 = measured_renderer.cell_height.into();

                        let cols = ((available_width / cell_width_f32) as usize).max(1);
                        let rows = ((available_height / cell_height_f32) as usize).max(1);

                        // Helper struct implementing Dimensions for resize
                        struct TermSize {
                            cols: usize,
                            rows: usize,
                        }
                        impl Dimensions for TermSize {
                            fn total_lines(&self) -> usize {
                                self.rows
                            }
                            fn screen_lines(&self) -> usize {
                                self.rows
                            }
                            fn columns(&self) -> usize {
                                self.cols
                            }
                            fn last_column(&self) -> alacritty_terminal::index::Column {
                                alacritty_terminal::index::Column(self.cols.saturating_sub(1))
                            }
                            fn bottommost_line(&self) -> alacritty_terminal::index::Line {
                                alacritty_terminal::index::Line(self.rows as i32 - 1)
                            }
                            fn topmost_line(&self) -> alacritty_terminal::index::Line {
                                alacritty_terminal::index::Line(0)
                            }
                        }

                        // Resize terminal if dimensions changed
                        let mut term = state_arc.lock();
                        let current_cols = term.columns();
                        let current_rows = term.screen_lines();
                        if cols != current_cols || rows != current_rows {
                            // Notify the PTY about the resize
                            if let Some(ref callback) = resize_callback {
                                callback(cols, rows);
                            }
                            term.resize(TermSize { cols, rows });
                        }

                        // Paint the terminal with measured dimensions
                        measured_renderer.paint(bounds, padding, &term, window, cx);
                    },
                )
                .size_full(),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::point;

    fn geometry() -> TerminalGeometry {
        TerminalGeometry {
            origin: point(px(10.0), px(20.0)),
            cell_width: px(10.0),
            cell_height: px(20.0),
        }
    }

    #[test]
    fn hit_cell_uses_cell_half_and_visible_row() {
        assert_eq!(
            hit_cell(point(px(34.0), px(45.0)), geometry(), 8, 4, 0),
            (AlacPoint::new(Line(1), Column(2)), Side::Left)
        );
        assert_eq!(
            hit_cell(point(px(36.0), px(45.0)), geometry(), 8, 4, 0),
            (AlacPoint::new(Line(1), Column(2)), Side::Right)
        );
    }

    #[test]
    fn hit_cell_clamps_outside_grid() {
        assert_eq!(
            hit_cell(point(px(34.0), px(10.0)), geometry(), 8, 4, 0),
            (AlacPoint::new(Line(0), Column(2)), Side::Left)
        );
        assert_eq!(
            hit_cell(point(px(5.0), px(45.0)), geometry(), 8, 4, 0),
            (AlacPoint::new(Line(1), Column(0)), Side::Left)
        );
        assert_eq!(
            hit_cell(point(px(100.0), px(45.0)), geometry(), 8, 4, 0),
            (AlacPoint::new(Line(1), Column(7)), Side::Right)
        );
        assert_eq!(
            hit_cell(point(px(34.0), px(105.0)), geometry(), 8, 4, 0),
            (AlacPoint::new(Line(3), Column(7)), Side::Right)
        );
    }

    #[test]
    fn hit_cell_maps_visible_rows_into_scrollback() {
        assert_eq!(
            hit_cell(point(px(11.0), px(21.0)), geometry(), 8, 4, 5),
            (AlacPoint::new(Line(-5), Column(0)), Side::Left)
        );
    }

    #[test]
    fn selection_in_scrollback_follows_new_output() {
        let (tx, _rx) = mpsc::channel();
        let mut state = TerminalState::new_with_scrollback(8, 3, 10, GpuiEventProxy::new(tx));
        state.process_bytes(b"alpha\r\nbravo\r\ncharlie\r\n");
        state.with_term_mut(|term| {
            term.scroll_display(Scroll::Delta(1));
            let offset = term.grid().display_offset();
            let (start, side) = hit_cell(point(px(11.0), px(21.0)), geometry(), 8, 3, offset);
            let (end, end_side) = hit_cell(point(px(59.0), px(21.0)), geometry(), 8, 3, offset);
            let mut selection = Selection::new(SelectionType::Simple, start, side);
            selection.update(end, end_side);
            term.selection = Some(selection);
            assert_eq!(term.selection_to_string().as_deref(), Some("alpha"));
        });

        state.process_bytes(b"delta\r\n");
        state.with_term(|term| assert_eq!(term.selection_to_string().as_deref(), Some("alpha")));
    }
}
