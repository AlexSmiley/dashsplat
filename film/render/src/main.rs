use fframes::cli::clap;
use fframes::{AudioMixOptions, StaticMediaProvider, CombinedMediaProvider, EncoderOptions, MediaDirectory, MediaProvider, RenderOptions, cli};
use fframes_skia_renderer::{SkiaFFramesRenderer, SkiaPipelineConcurrencyPolicy, SkiaPipelineConfig, metal::SkiaMetalCtx};
use journey::{FilmFonts, HEIGHT, Journey, WIDTH};
use std::process::ExitCode;

#[derive(Debug, clap::Args)]
struct VideoArgs {}

fn main() -> ExitCode {
    let args = cli::parse::<VideoArgs>();
    // CHAPTER=n renders one chapter on its own (it then opens and closes cleanly); STEM=amb|sfx picks a sound stem.
    let chapter = std::env::var("CHAPTER").ok().and_then(|c| c.parse::<usize>().ok());
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let fonts = FilmFonts::prepare().expect("fonts");
    let media_dir = MediaDirectory::read_folder(format!("{root}/work/media")).expect("film/work/media (run conform.py)");
    let media = media_dir.process_media_source().expect("media");
    std::fs::create_dir_all(format!("{root}/work/audio")).ok();
    let audio_dir = MediaDirectory::read_folder(format!("{root}/work/audio")).expect("film/work/audio");
    let audio = audio_dir.process_media_source().expect("audio");
    let all = CombinedMediaProvider::from([&fonts as &dyn MediaProvider, &media, &audio]);
    let video = Journey::new(chapter);
    let gpu = SkiaMetalCtx::new(WIDTH, HEIGHT).expect("GPU context");
    cli::new(
        &video,
        RenderOptions {
            media: Some(&all),
            default_font: "Berkeley Mono",
            // raw mix without the master limiter; loudness and peaks are set when the stems are delivered
            audio_mix: AudioMixOptions { limiter: None, ..Default::default() },
            video_encoder_options: EncoderOptions {
                preferred_encoder: Some("libx264"),
                codec_params: Some(&[("crf", "19"), ("preset", "medium"), ("tune", "grain")]),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .args(args)
    .backend(
        SkiaFFramesRenderer::new_metal(&gpu, SkiaPipelineConfig { concurrency_policy: SkiaPipelineConcurrencyPolicy::MaxPerformance, ..Default::default() })
            .expect("skia renderer"),
    )
    .preview(fframes_native_player::cli_preview)
    .default_output("../out/journey-raw.mp4")
    .run()
}
