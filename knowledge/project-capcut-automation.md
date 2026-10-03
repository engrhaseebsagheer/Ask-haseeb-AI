# CapCut Automation

CapCut Automation is a Python tool by Haseeb Sagheer that places images, photographer credits and animations into CapCut projects automatically, for long-form YouTube videos.

Type: Automation tool. Status: Private tool. Built by Haseeb Sagheer, solo.
What the status means: Private tool means I use it in my own editing workflow. It is not published as an app or a repository. I can show it working on a call, or adapt it to the way your videos are put together.
Portfolio page: https://haseebsagheer.com/projects/capcut-automation/

## Overview
CapCut Automation prepares a CapCut project before I start editing. It places the images, attaches the right photographer credit to each one, and applies the animations.

Long documentary-style videos are where it pays off. A single video can need dozens of images, and each one needs a credit on screen. Doing that by hand means dragging, typing and aligning the same thing over and over, and it is easy to attach a credit to the wrong image.

The output is an ordinary CapCut project. I open it, adjust what needs adjusting, and export. The tool handles the repetitive placement and leaves the editorial decisions to me.

## The problem CapCut Automation solves
A long documentary-style video uses dozens of images, and each one needs a photo credit and an animation. Dragging them into the timeline one at a time is slow and easy to get wrong.

## Who CapCut Automation is for
Editors of long, image-heavy videos, where every image needs a credit and an animation.

## What CapCut Automation does
- Places images into the CapCut project
- Adds the photographer credit that belongs to each image
- Applies animations to the placed images
- Leaves a normal CapCut project that can still be edited by hand

## How CapCut Automation works, step by step
1. Inputs: Images and their credits are prepared for the video.
2. Write: The tool writes them into the CapCut project.
3. Animate: Animations are applied to each placed image.
4. Open: The project opens in CapCut ready for final touches.

## Engineering notes for CapCut Automation
- Works on the project itself: The tool writes into the CapCut project, so the result opens in CapCut like any other project.
- Credits stay attached: Each photographer credit is placed with the image it belongs to.
- Still editable: Nothing is locked. The editor can move, trim or replace anything afterwards.

## Built with
Python, CapCut project files

## Haseeb's role on CapCut Automation
I wrote the tool in Python and use it on my own long-form videos.

## Questions about CapCut Automation
### What does CapCut Automation do?
It places images, photo credits and animations into a CapCut project automatically, so the editor starts from a nearly finished timeline.

### Is it separate from Video Maker?
Yes. Video Maker renders a video from a script on its own. CapCut Automation prepares a CapCut project that is then finished in CapCut.

### Does it replace the editor?
No. It removes the repetitive placement work. The final edit is still done by a person in CapCut.

### What is it written in?
Python.

