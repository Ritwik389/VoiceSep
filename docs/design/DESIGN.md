# VoiceSep workspace direction

Working screen, not a marketing landing page. Approved palette and layout take precedence over generic design anchors.

Primary reference: [ElevenLabs design analysis](https://raw.githubusercontent.com/voltagent/awesome-design-md/main/design-md/elevenlabs/DESIGN.md), consulted for restrained voice-tool styling. The [Linear alternative](https://raw.githubusercontent.com/voltagent/awesome-design-md/main/design-md/linear.app/DESIGN.md) was considered for density; its project-management composition is less suitable than an audio timeline.

Generated design reference: `workspace-reference.png`. It establishes a slim sessions rail, wide track canvas, synchronized transport, and compact inspector. Analysis: charcoal surfaces are separated by tone; a 168px label column aligns tracks; waveform colors belong to speaker data; small controls remain readable; selected-speaker listening is the signature action. The implementation adapts its example screen to an honest empty/imported state because no processed conversations exist yet. Generated example names/transcripts are not product data or bundled separation results.

Adapted tokens: ground #111517, surfaces #181d20/#252d31, text #edf1f2, action #85e3df, muted text #a0adb3; speaker colors blue/amber/rose/green/tan. System sans typography, flat panels, 6px fields, 10px major surfaces. Three columns at desktop; details move below the editor on tablets; sessions become a horizontal strip on phones. Use daisyUI semantics, native controls, visible focus, and reduced-motion behavior. No decorative photos, fake model scores, or unimplemented speaker-trait panels.
