# Media handoff

Drag
[`vLLM-Omni Media Handoff.json`](../example_workflows/vLLM-Omni%20Media%20Handoff.json)
onto the canvas. It connects a saved H3 or restored video to an external
soundtrack using built-in ComfyUI nodes. Choose your files in **Load Video** and
**Load Audio**; the example filenames are placeholders in ComfyUI's input folder.

Queue once to save four outputs under `output/CUI03/`:

| Output | Audio policy |
| --- | --- |
| `preserve` | Save the original VIDEO with its original audio. |
| `replace` | Use only the supplied soundtrack with the source video frames. |
| `mix` | Use AudioMerge's `mean`: average the source and supplied waveforms. |
| `repeat-twice` | Join two copies of the supplied soundtrack, then replace the source audio. |

**Save Video** clips audio longer than the video. A shorter replacement or
repeated soundtrack ends early; the remaining video has no soundtrack.
`repeat-twice` makes exactly two copies with a hard join, without a crossfade or
automatic repetition until the end of the video. Use **Audio Adjust Volume**
before **AudioMerge** when you need a different balance.

The rebuilt branches receive FPS, bit depth and color space from
**Get Video Components**, rather than using a separate FPS setting. They
re-encode the source frames; they do not promise byte-identical video output.
Replacement preserves the supplied audio's sample rate and channel count.
AudioConcat converts a mono soundtrack to stereo by duplicating its channel.
Mixing uses the standard AudioMerge sample-rate conversion and normalization;
a mono supplied track broadcasts across an H3 stereo source. The source-video
branch remains separate, so changing the soundtrack does not discard the
original output.

## Connect remote services

To process a fresh generation, connect **Generate Video → VIDEO** to both
**Get Video Components → video** and **Save original → video**, replacing
**Load Video**. The same sockets accept a restoration node's VIDEO output.
An AUDIO returned by **Generate Music** may replace the Load Audio output.
Each generation/restoration node must be configured for its own service URL
and served model. The media nodes operate on returned media; they do not load
or switch remote model weights.

H3 generation setup is covered by the [MiniMax-H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md).
This saved-media template can be validated on CPU after generation.

## Compatibility and validation

Use a ComfyUI version that exposes the five Get Video Components outputs,
including `bit_depth` and `color_space`. The validation baseline is ComfyUI
`06c67806bcb36789a1e8e1278c6a764b389ff858`. No additional custom media nodes
are required.

After running the graph, inspect each MP4:

```bash
ffprobe -v error -count_frames \
  -show_entries stream=codec_type,nb_read_frames,r_frame_rate,time_base,duration,sample_rate,channels \
  -of json output/CUI03/replace_00001_.mp4
```

Compare video frame count and FPS with the source; check the intended audio
sample rate/channel count, listen to the selected audio policy, and watch for
audio/video timing differences. Test both a longer soundtrack and a shorter
one so clipping and early audio termination are visible. Record the source H3
server/model revision, source file, ComfyUI revision and graph revision with
the results. Workflow-schema checks alone do not establish media correctness.
