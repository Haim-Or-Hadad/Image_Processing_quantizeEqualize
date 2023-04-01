from  ex1_main import transformYIQ2RGB,imReadAndConvert,imDisplay,transformRGB2YIQ


# imDisplay('bac_con.png',1)

image=imReadAndConvert('beach.jpg',0)
yiq_img=transformYIQ2RGB(image)