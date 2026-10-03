# DoseAnchor

DoseAnchor is an Android medication reminder app for families, built by Haseeb Sagheer with React Native (Expo) and Supabase. It reminds the person taking the medication and alerts a caregiver. It is in closed testing on Google Play.

Type: Product. Status: Closed testing. Built by Haseeb Sagheer, solo.
What the status means: Closed testing means the Android app is on Google Play for a limited group of testers before public release. The website at doseanchor.com is taking waitlist sign-ups in the meantime.
Links: https://doseanchor.com
Portfolio page: https://haseebsagheer.com/projects/doseanchor/

## Overview
DoseAnchor is a medication reminder app built for families. The parent or relative gets a gentle reminder. The family sees which doses were taken.

The gap it fills is the phone call. Many people check on a parent's medication by ringing every morning to ask. With DoseAnchor, a follow-up reminder goes out if a dose is not marked as taken, and if one slips the family is told.

I built the app in React Native with Expo on a Supabase backend, set up subscriptions through Google Play billing, and built the website. It is a reminder tool, not medical advice.

## The problem DoseAnchor solves
A reminder only helps the person holding the phone. Families looking after a parent or relative need to know when a dose was missed.

## Who DoseAnchor is for
Family caregivers who want to know a parent or relative took their medicine without calling to ask.

## What DoseAnchor does
- Sends medication reminders on schedule
- Alerts a caregiver about the doses they are watching
- Sends reminder and account emails from the app’s own domain
- Handles subscriptions through Google Play billing

## How DoseAnchor works, step by step
1. Schedule: Medications and times are set up once.
2. Remind: The app reminds the person at each dose.
3. Alert: A caregiver is told when something needs attention.

## Engineering notes for DoseAnchor
- Two sides: The person taking the medication gets the reminder. The family sees which doses were taken.
- Alerts: If a dose slips, the family is told. The family alert is a Premium feature.
- Billing: Subscriptions run through Google Play billing with RevenueCat.
- Email: Reminder and account emails are sent from the product’s own domain.
- Not medical advice: It is a reminder tool.

## Built with
React Native (Expo), Supabase, RevenueCat, Google Play billing

## Haseeb's role on DoseAnchor
I built the app in React Native with Expo, the Supabase backend, the billing and the website.

## What is next for DoseAnchor
- Public release on Google Play after closed testing

## Questions about DoseAnchor
### What is DoseAnchor?
An Android medication reminder app for families, with alerts for caregivers.

### Is DoseAnchor available yet?
It is in closed testing on Google Play. Android is the only platform for now.

### Do caregivers pay?
No. The website states that caregivers never pay.

### Is there a free plan?
A free plan is planned at launch, with a Premium tier for features such as the family alert.

### How do I get it?
Join the waitlist at doseanchor.com.

