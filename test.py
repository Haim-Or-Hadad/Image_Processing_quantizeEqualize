from  ex1_main import hsitogramEqualize,transformYIQ2RGB,imReadAndConvert,imDisplay,transformRGB2YIQ


# imDisplay('bac_con.png',1)

image=imReadAndConvert('beach.jpg',1)
yiq_img=transformYIQ2RGB(image)


img = imReadAndConvert('beach.jpg',0)
hsitogramEqualize(img)