# hyperplane-google-winter-camp


## Space Travel

- There are times that you want to travel to some specific scenic spots but is drawn back by the long distance and high expense. 
- You may also want to know what your self image looks like in anime style.
- Then this Space Travel program can help realize your dream! All you need to do is updating a selfie and choosing the specific place you want to go.

![avatar](web_server/static/img/readme.jpg)
## Instructions
This repo contains three deep models, which do the human face transfer, background transfer and image matting respectively.

Please run the `app.py` in the `web_server` folder to start the program, and try the demo by visiting: http://localhost:8181

All dependency is contained in the `requirements.yml` file. Just enjoy it !!!

## Demos

Here are some demo images and demo videos.

---

- Background Style Transfer (Artist Style and Anime Style is shown below.)

![ava](web_server/static/img/background.jpg)

---
- Human Matting and Face Style Translation

![ava](web_server/static/img/matting.jpg)

---
- Foreground and Background Fusion and Final Result. 

![ava](web_server/static/img/result.jpg)

---

- Html5 Demo &  Real Time Human Matting and Background Style Transfer in Video Data.

<video id="video" controls="" preload="none" poster="http://om2bks7xs.bkt.clouddn.com/2017-08-26-Markdown-Advance-Video.jpg">
<source id="mp4" src="http://om2bks7xs.bkt.clouddn.com/2017-08-26-Markdown-Advance-Video.mp4" type="video/mp4">
</video>
<table align='center'>
<tr align='center'>
<td> Html5 Demo </td>
<td> Real Time Matting </td>
</tr>
<tr>
<td><img src = 'web_server/static/img/html5.jpg'>
<td><img src = 'web_server/static/img/video.jpg'>
</tr>
</table>
 
## Reference

[1] Kim J, Kim M, Kang H, et al. U-GAT-IT

[2] Li Y, Fang C, Yang J, et al. Universal style transfer via feature transforms

[3] Zhu, Bingke et al. “Fast Deep Matting for Portrait Animation on Mobile Phone.” 

[4] https://github.com/taki0112/UGATIT

[5] https://github.com/Yijunmaverick/UniversalStyleTransfer

[6] https://github.com/ofirlevy/FastMattingPortrait

