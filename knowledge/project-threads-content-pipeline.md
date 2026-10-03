# Threads Content Pipeline

Threads Content Pipeline is a Python tool by Haseeb Sagheer that analyses a Threads profile each week, drafts posts from what it learns, and sends every draft to a Telegram bot for review before anything is posted.

Type: Automation tool. Status: Private tool. Built by Haseeb Sagheer, solo.
What the status means: Private tool means I built it for my own account and it is not offered as a product. The code is not public. It is here because it shows how I build automation that keeps a person in control of what gets published.
Portfolio page: https://haseebsagheer.com/projects/threads-content-pipeline/

## Overview
This pipeline keeps a Threads account active without turning it over to a bot. It studies the profile, drafts posts, and then waits for a person to approve each one.

Analysis runs weekly. The first run reads the whole profile. Every run after that reads only the last week of new posts, so it stays quick and reflects what is working now.

Drafts go to a Telegram bot. Each one can be approved as written or edited first. Nothing is posted until that happens, which is the difference between this and a blind auto-poster.

## The problem Threads Content Pipeline solves
Posting consistently takes time, and a fully automatic bot posts things nobody checked. This keeps the drafting automatic and the final decision human.

## Who Threads Content Pipeline is for
Someone who wants to post consistently on Threads without handing the final say to a bot.

## What Threads Content Pipeline does
- Runs a full profile analysis on first run
- Re-analyses only the last week of new posts after that
- Drafts posts based on the analysis
- Sends each draft to a Telegram bot to approve or edit
- Posts only what was approved

## How Threads Content Pipeline works, step by step
1. Analyse: A weekly profile analysis looks at recent posts.
2. Draft: New posts are drafted from that analysis.
3. Review: Drafts arrive in Telegram, where they are approved or edited.
4. Post: Approved posts go out.

## Engineering notes for Threads Content Pipeline
- Weekly analysis: The first run analyses the whole profile. After that, each weekly run looks only at the last week of new posts.
- Human review: Drafts are delivered to a Telegram bot, where each one is approved or edited before it is posted.
- Personal tool: It was built for my own use, not as a product.

## Built with
Python, Telegram bot, Profile analysis

## Haseeb's role on Threads Content Pipeline
I built the analysis module, the drafting step and the Telegram review flow in Python.

## Questions about Threads Content Pipeline
### Does the pipeline post without review?
No. Every draft goes to a Telegram bot first, where it is approved or edited before posting.

### How often does it analyse the profile?
Weekly. The first run is a full analysis; later runs look only at the last week of new posts.

### Why Telegram for review?
It puts the approve or edit decision on a phone, so drafts can be reviewed anywhere.

### Is it a product?
No. It is a personal tool.

