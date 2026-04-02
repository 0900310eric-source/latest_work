架構:

播放主機 --> 本地瀏覽器 --> 另一台主機

播放主機:

1. 找開源影片來下載 (找多種畫質)
2. 影片拿來本地切成 H.264 DASH 格式
3. 打開 windows powershell (退到 offline_dash 資料夾外輸入) > python -m http.server 8000 (python 內建 server 程式, 未來要擴充 HTTP(3) 要自己寫)

本地瀏覽器:

1. ****** 使用 player_v2.html 來操控 (播放不同影片的時候, 更換 value="/movie1/stream.mpd" 即可, 目前的分類是六種) *****
2. ***** 記得檢查 IP *****

另一台主機:

1. 先開 BASH 限制流量程式 (播完影片按 Enter 拿到 PCAP files, 然後直接用 WireShark 輸出 csv file 畫圖)
2. 輸入網站網址
3. 播放影片
4. 按網站上 download csv 下載應用層資料