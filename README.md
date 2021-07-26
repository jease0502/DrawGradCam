# DrawGradCam

繪製 GradCam 

## 使用方式

 更改 main.py 裡面的參數

- model_path 模型位置 ex. ???/model.h5
- image_path 圖檔位置 ex./nfs/img/
- draw_image 要繪製的那張圖的位置 ex.10462.jpg
- output_file 圖片輸出位置 ex. ???/

如果一次要畫很多張圖，輸入為資料夾，增加 mode 這個參數，然後 draw_image 改成輸入的圖片資料夾位置
- draw_image ex. ???/
- mode = 1

如果一次要畫很多張圖，輸入為txt，增加 mode 這個參數，然後 draw_image 改成輸入的txt位置
- draw_image ex. ???.txt
- mode = 2

