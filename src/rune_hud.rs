//! The dictation HUD on Linux and Windows: the macOS HUD's shaders, ported to
//! WGSL in `rune_hud.wgsl`, drawn on the GPU by a Rune overlay's compositor.
//! The overlay is focus-free and click-through: a layer-shell surface on
//! Wayland, an override-redirect window on X11, and a topmost tool window
//! that never activates on Windows.
//!
//! A HUD thread owns the shared indicator model and, with the `dot-scope`
//! experiment on, Dot scope's simulation. It drains the listener's events,
//! steps both every 16 ms while the HUD is on screen (sleeping otherwise),
//! and hands each frame's uniforms to the overlay, whose compositor only
//! draws them. The listener never waits on it.
//!
//! The Sounds sheet draws its preview with the same [`Painter`], frames and
//! [`DotScope`] on every OS.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rune::compose::{Compositor, Frame};
use rune::{Cx, El, OverlayHandle, OverlayOptions, OverlaySpec, ScreenEdge, el};

use crate::app_settings::IndicatorPosition;
use crate::desktop_indicator_model::{
    DictationIndicatorEvent, IndicatorModel, IndicatorUniforms, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use crate::dot_scope::{self, MAX_DOTS, Pose, Scope, Stage, Tuning};

pub(crate) use crate::desktop_indicator_model::DictationIndicatorEvent as HudEvent;

const FRAME: Duration = Duration::from_millis(16);
pub(crate) const TOP_OFFSET: f32 = crate::desktop_indicator_model::TOP_OFFSET as f32;
pub(crate) const BOTTOM_OFFSET: f32 = 24.0;
/// Dot scope's window: room for its widest and tallest plate and the drop
/// shadow below it.
pub(crate) const DOT_SCOPE_WINDOW: (f32, f32) = (176.0, 80.0);

fn dot_scope_enabled() -> bool {
    crate::experiments::enabled(crate::experiments::Experiment::DotScope)
}

/// How the listener reports captures to the HUD; sending never waits.
#[derive(Clone)]
pub(crate) struct HudSender {
    events: Sender<HudEvent>,
    position: Arc<Mutex<IndicatorPosition>>,
}

impl HudSender {
    pub(crate) fn send(&self, event: HudEvent) {
        let _ = self.events.send(event);
    }

    pub(crate) fn meter(&self, samples: &[f32]) {
        if let Some(event) = DictationIndicatorEvent::meter(samples) {
            self.send(event);
        }
    }

    pub(crate) fn set_position(&self, position: IndicatorPosition) {
        *self
            .position
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = position;
    }
}

/// One frame for the compositor.
pub(crate) enum Drawn {
    Capsule(IndicatorUniforms),
    Dots(DotFrame),
}

impl Drawn {
    /// This frame faded by `opacity`, as an enclosing view fades.
    pub(crate) fn faded(mut self, opacity: f32) -> Self {
        match &mut self {
            Drawn::Capsule(uniforms) => uniforms.opacity *= opacity,
            Drawn::Dots(frame) => frame.pose.opacity *= opacity,
        }
        self
    }
}

pub(crate) struct DotFrame {
    pub(crate) pose: Pose,
    tuning: Tuning,
    columns: usize,
    rows: usize,
    /// (brightness, freshness) per dot, row by row, as normalized u16s.
    dots: Vec<[u16; 2]>,
}

#[derive(Default)]
pub(crate) struct Hud {
    drawn: Option<Drawn>,
}

/// The overlay; [`HudHandle::start`] starts the HUD thread that feeds it.
pub(crate) struct HudHandle {
    overlay: OverlayHandle<Hud>,
}

impl HudHandle {
    pub(crate) fn start(self, initial: IndicatorPosition) -> HudSender {
        let (events, receiver) = mpsc::channel();
        let position = Arc::new(Mutex::new(initial));
        let hud_position = position.clone();
        let _ = std::thread::Builder::new()
            .name("hex-hud".into())
            .spawn(move || {
                run(self.overlay, receiver, || {
                    *hud_position
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                })
            });
        HudSender { events, position }
    }

    /// The overlay, to ask later whether the platform could show it.
    #[cfg(target_os = "linux")]
    pub(crate) fn overlay(&self) -> OverlayHandle<Hud> {
        self.overlay.clone()
    }
}

/// Adds the overlay to the window's app.
pub(crate) fn install<S: 'static>(app: &mut rune::App<S>) -> HudHandle {
    let size = if dot_scope_enabled() {
        DOT_SCOPE_WINDOW
    } else {
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    };
    let overlay = app.overlay(
        OverlaySpec {
            name: "hex-hud",
            size,
            edge: ScreenEdge::Top,
            margin: TOP_OFFSET,
            align: rune::Align::Center,
            inset: 0.0,
            interactive: false,
            view,
        }
        .options(OverlayOptions::default())
        // The shaders write display (sRGB-encoded) values, as on macOS.
        .srgb_blend(true)
        .compositor(HudCompositor::default()),
        Hud::default(),
    );
    HudHandle { overlay }
}

/// The HUD thread: returns when the listener's event channel closes.
fn run(
    overlay: OverlayHandle<Hud>,
    events: Receiver<HudEvent>,
    position: impl Fn() -> IndicatorPosition,
) {
    let mut model = IndicatorModel::new();
    let mut scope = dot_scope_enabled().then(|| DotScope::new(Tuning::load()));
    let mut last_frame = Instant::now();
    let mut shown: Option<IndicatorPosition> = None;
    loop {
        let event = if shown.is_some() {
            events.recv_timeout(FRAME)
        } else {
            events.recv().map_err(|_| RecvTimeoutError::Disconnected)
        };
        let mut handle = |event: HudEvent| {
            if let Some(scope) = &mut scope
                && matches!(event, HudEvent::Started | HudEvent::EditingStarted)
            {
                // HUD Lab's saved tuning applies from the next dictation.
                scope.restart(Tuning::load());
            }
            model.handle(event);
        };
        match event {
            Ok(event) => {
                handle(event);
                while let Ok(event) = events.try_recv() {
                    handle(event);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        let now = Instant::now();
        let dt = now
            .duration_since(last_frame)
            .as_secs_f32()
            .clamp(1.0 / 240.0, 1.0 / 20.0);
        last_frame = now;
        let frame = model.frame(now);
        let (drawn, on_screen) = match &mut scope {
            Some(scope) => {
                let (stage, visible, loudness) = model.dot_scope(now);
                scope.advance(dt, stage, visible, loudness);
                (Drawn::Dots(scope.frame()), visible || !scope.settled())
            }
            None => (Drawn::Capsule(frame.uniforms), frame.visible),
        };
        let position = position();
        if !on_screen || position == IndicatorPosition::Hidden {
            if shown.take().is_some() {
                overlay.hide();
            }
            continue;
        }
        overlay.update(move |hud, _| hud.drawn = Some(drawn));
        if shown != Some(position) {
            let (edge, margin) = match position {
                IndicatorPosition::Bottom => (ScreenEdge::Bottom, BOTTOM_OFFSET),
                _ => (ScreenEdge::Top, TOP_OFFSET),
            };
            overlay.place(edge, margin);
            overlay.show();
            shown = Some(position);
        }
    }
    overlay.hide();
}

/// Dot scope's simulation and the stage it last showed.
pub(crate) struct DotScope {
    scope: Scope,
    stage: Stage,
}

impl DotScope {
    pub(crate) fn new(tuning: Tuning) -> Self {
        Self {
            scope: Scope::new(tuning),
            stage: Stage::Recording,
        }
    }

    pub(crate) fn restart(&mut self, tuning: Tuning) {
        self.scope.tune(tuning);
        self.stage = Stage::Recording;
        self.scope.restart(false);
    }

    /// `stage` is `None` while the HUD hides or a capture was cancelled:
    /// the last stage fades out.
    pub(crate) fn advance(&mut self, dt: f32, stage: Option<Stage>, visible: bool, loudness: f32) {
        if let Some(stage) = stage {
            self.stage = stage;
        }
        self.scope.advance(
            dt,
            dot_scope::Frame {
                stage: self.stage,
                visible,
                loudness,
                reduce_motion: false,
            },
        );
    }

    /// Whether it has faded out.
    pub(crate) fn settled(&self) -> bool {
        self.scope.settled()
    }

    pub(crate) fn frame(&self) -> DotFrame {
        DotFrame {
            pose: self.scope.pose(),
            tuning: self.scope.tuning(),
            columns: self.scope.columns(),
            rows: self.scope.rows(),
            dots: self
                .scope
                .dots()
                .iter()
                .map(|value| value.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16))
                .collect(),
        }
    }
}

fn view(_: &Hud, _: &mut Cx<Hud>) -> El {
    el().key("hud").w_full().h_full()
}

/// `DotUniforms` in `rune_hud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DotUniforms {
    resolution: [f32; 2],
    backing_scale: f32,
    width: f32,
    height: f32,
    opacity: f32,
    scale: f32,
    softness: f32,
    pitch: f32,
    columns: f32,
    rows: f32,
    glow: f32,
    grid: f32,
    live: f32,
    cool: f32,
    dark: f32,
    base: [f32; 4],
    core: [f32; 4],
    plate: [f32; 4],
    rim: [f32; 4],
    shadow: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<DotUniforms>() == 144);
const _: () = assert!(std::mem::size_of::<IndicatorUniforms>() == 112);
const DOT_BYTES: u64 = (MAX_DOTS * 4) as u64;

fn bytes<T: Copy>(value: &T) -> &[u8] {
    // SAFETY: `T` is a `repr(C)` struct of `f32`s with no padding, so every
    // byte is initialized.
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), std::mem::size_of::<T>()) }
}

#[derive(Default)]
struct HudCompositor {
    gpu: Option<Painter>,
}

impl Compositor<Hud> for HudCompositor {
    fn render(&mut self, f: &mut Frame<'_, Hud>) {
        f.draw_ui();
        let Some(drawn) = &f.rt.state.drawn else {
            return;
        };
        if self.gpu.as_ref().is_none_or(|gpu| gpu.format != f.format) {
            self.gpu = Some(Painter::new(f.device, f.format));
        }
        let gpu = self.gpu.as_ref().expect("created above");
        let (width, height) = (f.size.0 as f32, f.size.1 as f32);
        gpu.draw(
            f.queue,
            f.encoder,
            f.target,
            drawn,
            Target {
                viewport: [0.0, 0.0, width, height],
                clip: [0, 0, f.size.0, f.size.1],
                scale: f.rt.env.scale,
                dark: f.rt.env.system_dark,
            },
        );
    }
}

/// What a frame draws into.
#[derive(Clone, Copy)]
pub(crate) struct Target {
    /// The HUD's window in the target, x, y, width and height in device
    /// pixels.
    pub(crate) viewport: [f32; 4],
    /// The pixels it may touch, x, y, width and height, inside the target.
    pub(crate) clip: [u32; 4],
    /// Device pixels per point.
    pub(crate) scale: f32,
    pub(crate) dark: bool,
}

/// The HUD's shaders, ready to draw frames into targets of one format.
pub(crate) struct Painter {
    pub(crate) format: wgpu::TextureFormat,
    capsule: wgpu::Buffer,
    dot_uniforms: wgpu::Buffer,
    dots: wgpu::Buffer,
    bind: wgpu::BindGroup,
    capsule_pipeline: wgpu::RenderPipeline,
    dot_pipeline: wgpu::RenderPipeline,
}

impl Painter {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Painter {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hex hud"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rune_hud.wgsl").into()),
        });
        let buffer = |label, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let capsule = buffer(
            "hex hud capsule",
            std::mem::size_of::<IndicatorUniforms>() as u64,
        );
        let dot_uniforms = buffer(
            "hex hud dot uniforms",
            std::mem::size_of::<DotUniforms>() as u64,
        );
        let dots = buffer("hex hud dots", DOT_BYTES);
        let entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hex hud"),
            entries: &[entry(0), entry(1), entry(2)],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hex hud"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: capsule.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dot_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dots.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hex hud"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // Display values; a linear target decodes them.
        let constants = [("linear_target", f64::from(u8::from(format.is_srgb())))];
        let pipeline = |fragment: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("indicator_vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fragment),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        Painter {
            format,
            capsule,
            dot_uniforms,
            dots,
            bind,
            capsule_pipeline: pipeline("indicator_fragment"),
            dot_pipeline: pipeline("dot_scope_fragment"),
        }
    }

    /// Draws `drawn` over `view`, which holds what is under it.
    pub(crate) fn draw(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        drawn: &Drawn,
        target: Target,
    ) {
        let [x, y, width, height] = target.viewport;
        let resolution = [width, height];
        let pipeline = match drawn {
            Drawn::Capsule(uniforms) => {
                let uniforms = IndicatorUniforms {
                    resolution,
                    ..*uniforms
                };
                queue.write_buffer(&self.capsule, 0, bytes(&uniforms));
                &self.capsule_pipeline
            }
            Drawn::Dots(frame) => {
                let dark = target.dark;
                let colors = dot_scope::palette(dark, &frame.pose, frame.tuning.saturation);
                let rgba = |c: [f32; 3]| [c[0], c[1], c[2], 1.0];
                let pose = &frame.pose;
                let uniforms = DotUniforms {
                    resolution,
                    backing_scale: target.scale,
                    width: pose.width,
                    height: pose.height,
                    opacity: pose.opacity,
                    scale: pose.scale,
                    softness: pose.softness,
                    pitch: frame.tuning.spacing,
                    columns: frame.columns as f32,
                    rows: frame.rows as f32,
                    glow: pose.glow,
                    grid: pose.grid,
                    live: pose.live,
                    cool: pose.cool,
                    dark: if dark { 1.0 } else { 0.0 },
                    base: rgba(colors.base),
                    core: rgba(colors.core),
                    plate: colors.plate,
                    rim: colors.rim,
                    shadow: colors.shadow,
                };
                queue.write_buffer(&self.dot_uniforms, 0, bytes(&uniforms));
                let packed: Vec<u8> = frame
                    .dots
                    .iter()
                    .take(MAX_DOTS)
                    .flat_map(|[power, fresh]| {
                        (u32::from(*power) | (u32::from(*fresh) << 16)).to_ne_bytes()
                    })
                    .collect();
                queue.write_buffer(&self.dots, 0, &packed);
                &self.dot_pipeline
            }
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hex hud"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(x, y, width, height, 0.0, 1.0);
        let [cx, cy, cw, ch] = target.clip;
        pass.set_scissor_rect(cx, cy, cw, ch);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_uniform_layout_matches_wgsl() {
        assert_eq!(std::mem::offset_of!(DotUniforms, dark), 60);
        assert_eq!(std::mem::offset_of!(DotUniforms, base), 64);
        assert_eq!(std::mem::offset_of!(DotUniforms, shadow), 128);
    }

    #[test]
    fn dot_scope_window_fits_the_largest_plate() {
        let largest = |key| {
            dot_scope::KNOBS
                .iter()
                .find(|knob| knob.key == key)
                .map(|knob| knob.max)
                .unwrap()
        };
        assert!(
            largest("dot-width").max(largest("dot-processing-width")) + 24.0 <= DOT_SCOPE_WINDOW.0
        );
        // The plate's middle sits in the window's; its shadow reaches about
        // 16 pt below it.
        assert!(largest("dot-height") / 2.0 + 19.0 <= DOT_SCOPE_WINDOW.1 / 2.0);
    }

    /// Renders a dictation through the real shaders, offscreen, as PNG
    /// frames at 60 fps: HUD Lab's synthetic voice for 3 s, then
    /// transcribing, processing, and done. Compare them with the macOS
    /// HUD's `dot_scope_frames`:
    ///
    /// ```sh
    /// HEX_HUD_FRAMES=/tmp/frames HEX_HUD_DOT_SCOPE=1 \
    ///   cargo test --bin hex rune_hud_frames -- --ignored
    /// ```
    ///
    /// `HEX_HUD_DOT_SCOPE=1` draws Dot scope (its saved tuning), otherwise
    /// the capsule; `HEX_HUD_THEME=light` a light desktop.
    #[test]
    #[ignore = "writes frames for a review clip"]
    fn rune_hud_frames() {
        let out =
            std::path::PathBuf::from(std::env::var("HEX_HUD_FRAMES").expect("HEX_HUD_FRAMES"));
        std::fs::create_dir_all(&out).unwrap();
        let dots = std::env::var("HEX_HUD_DOT_SCOPE").as_deref() == Ok("1");
        let dark = std::env::var("HEX_HUD_THEME").as_deref() != Ok("light");

        // The HUD thread's steps, in real time: the model reads the clock.
        let mut model = IndicatorModel::new();
        let mut scope = dots.then(|| DotScope::new(Tuning::load()));
        if let Some(scope) = &mut scope {
            scope.restart(Tuning::load());
        }
        model.handle(HudEvent::Started);
        let start = Instant::now();
        let mut frames = Vec::new();
        let mut events = vec![
            (3.0, HudEvent::Submitted { job_id: 1 }),
            (3.6, HudEvent::Processing { job_id: 1 }),
            (4.4, HudEvent::JobCompleted { job_id: 1 }),
        ];
        events.reverse();
        let mut last = start;
        for n in 0..330 {
            let due = start + Duration::from_secs_f32(n as f32 / 60.0);
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
            let t = due.duration_since(start).as_secs_f32();
            while events.last().is_some_and(|(at, _)| *at <= t) {
                model.handle(events.pop().unwrap().1);
            }
            if t < 3.0 {
                // A laptop microphone's RMS for the demo voice's loudness.
                let rms = 10_f32.powf(dot_scope::demo_loudness(t) / 20.0);
                model.handle(HudEvent::Meter {
                    average: rms,
                    peak: (rms * 3.0).min(1.0),
                });
            }
            let now = Instant::now();
            let dt = now
                .duration_since(last)
                .as_secs_f32()
                .clamp(1.0 / 240.0, 1.0 / 20.0);
            last = now;
            let frame = model.frame(now);
            frames.push(match &mut scope {
                Some(scope) => {
                    let (stage, visible, loudness) = model.dot_scope(now);
                    scope.advance(dt, stage, visible, loudness);
                    Drawn::Dots(scope.frame())
                }
                None => Drawn::Capsule(frame.uniforms),
            });
        }

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            block_on(instance.request_adapter(&Default::default())).expect("a GPU adapter");
        let (device, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let gpu = Painter::new(&device, format);
        let scale = 2.0;
        let window = if dots {
            DOT_SCOPE_WINDOW
        } else {
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        };
        let size = ((window.0 * scale) as u32, (window.1 * scale) as u32);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let row = (size.0 * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * size.1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let desktop: [f32; 3] = if dark {
            [0.11, 0.11, 0.13]
        } else {
            [0.93, 0.93, 0.94]
        };
        for (n, drawn) in frames.iter().enumerate() {
            let mut encoder = device.create_command_encoder(&Default::default());
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }));
            gpu.draw(
                &queue,
                &mut encoder,
                &view,
                drawn,
                Target {
                    viewport: [0.0, 0.0, size.0 as f32, size.1 as f32],
                    clip: [0, 0, size.0, size.1],
                    scale,
                    dark,
                },
            );
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| result.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let mapped = readback.slice(..).get_mapped_range().unwrap();
            // Premultiplied RGBA over the desktop, as RGB.
            let mut rgb = Vec::with_capacity((size.0 * size.1 * 3) as usize);
            for y in 0..size.1 {
                for x in 0..size.0 {
                    let p = &mapped[(y * row + x * 4) as usize..][..4];
                    let a = f32::from(p[3]) / 255.0;
                    for c in 0..3 {
                        let v = f32::from(p[c]) / 255.0 + desktop[c] * (1.0 - a);
                        rgb.push((v * 255.0).round().clamp(0.0, 255.0) as u8);
                    }
                }
            }
            drop(mapped);
            readback.unmap();
            let file = std::fs::File::create(out.join(format!("{n:04}.png"))).unwrap();
            let mut encoder = png::Encoder::new(file, size.0, size.1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&rgb)
                .unwrap();
        }
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                return output;
            }
            std::thread::yield_now();
        }
    }
}
