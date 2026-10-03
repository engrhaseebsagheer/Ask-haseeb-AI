# Video Maker

Video Maker is an automated pipeline built by Haseeb Sagheer that turns a text script into a finished video. It generates the visuals, places text and images on each frame without overlaps, animates the text, and mixes in sound effects.

Type: Automation tool. Status: Private tool. Built by Haseeb Sagheer, solo.
What the status means: Private tool means it runs on my own machine for my own videos. It is not a public app and the code is not published, so there is no link to try it. I can walk through it on a call, or build a version shaped around your content.
Portfolio page: https://haseebsagheer.com/projects/video-maker/

## Overview
Video Maker is a pipeline that produces a complete video from a script. I give it text, and it returns a rendered file with visuals, animated captions and sound.

It works in stages. First it generates an image for each part of the script and removes the background from each one. Then it lays out every frame: images and text are placed together, and each new element is checked against the ones already on the frame so nothing overlaps. Text is animated, infographic-style frames are built, sound effects are added, and the audio is mixed before the final render.

I built it because assembling these videos by hand is the same sequence every time. The parts that took the most engineering were the ones a human does without thinking: deciding where things go, and noticing when two things collide.

## The problem Video Maker solves
Producing one explainer video by hand means sourcing images, cutting out backgrounds, positioning every overlay, animating captions and laying sound effects. It is the same sequence every time, which makes it a job for a pipeline.

## Who Video Maker is for
Anyone who produces explainer-style videos on a schedule and is tired of assembling each one by hand.

## What Video Maker does
- Generates AI visuals for each part of the script
- Removes image backgrounds so overlays sit cleanly on the frame
- Places images and text with collision detection, so nothing overlaps
- Animates text overlays and builds infographic-style frames
- Adds sound effects and mixes the audio
- Renders the complete video with no manual editing step

## How Video Maker works, step by step
1. Script in: The pipeline takes text or a script as input.
2. Generate: Visual assets are generated for each scene.
3. Place: Text and images are positioned on the frame, checked against each other for overlap.
4. Mix: Sound effects are added and the audio is mixed.
5. Render: The finished video is written out.

## Engineering notes for Video Maker
- Placement: Images and text are placed with collision detection, so each overlay is checked against the others before the frame is rendered.
- Background removal: Generated images have their backgrounds removed so they sit on the frame as cut-outs instead of rectangles.
- Audio: Sound effects are added and mixed in the same run, so the output needs no separate audio pass.
- Recent work: The latest work went into image overlay rendering, background removal, and the combined text and image placement system.

## Built with
Python, AI image generation, Background removal, Collision-aware layout, Audio mixing

## Haseeb's role on Video Maker
I designed the pipeline and wrote all of it: generation, layout, animation, audio and rendering.

## Questions about Video Maker
### What does Video Maker do?
It takes a script and produces a complete video: generated visuals, placed and animated text, sound effects and the final render.

### How does it avoid overlapping text and images?
A placement step uses collision detection, so each overlay is positioned against the others before the frame is rendered.

### Is Video Maker available to use?
Not publicly. It is a private tool. If you want a similar pipeline for your own content, get in touch.

### Is it the same as CapCut Automation?
No. Video Maker renders the whole video itself. CapCut Automation prepares a CapCut project that is finished by hand in CapCut.

