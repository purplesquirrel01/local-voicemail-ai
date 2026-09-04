# Portal demonstration provenance

The portal image is recreated from the application's HTML template with a
hand-authored synthetic API response. It is not a production screenshot and
contains no cropped, blurred, or transformed production record.

All visible identities are fictional: reviewer Avery Example, caller Bailey
Sample, subject Jordan Sample, mailbox 9901, date 2 January 2000, arbitrary
birthdate 3 February 1970, and callback 202-555-0142. The transcript is a newly
written appointment request. Review/confidence labels are illustrative fixture
state, not a measured inference result. The audio control uses generated silence.

`tools/render_portal_demo.py` recreates the image with Playwright. Every browser
request is intercepted and answered from synthetic fixtures; no application
server, database, real recording, or network service is used. The script must
run in an isolated development workspace with the optional Playwright dependency
and a locally installed Chromium browser. Neither is required to run LVT.

The older `demo/portal-demo.png` path is retained with the same recreated image
so existing repository links continue to work.
