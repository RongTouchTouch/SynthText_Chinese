#coding:utf-8
import os
import pickle
import sys
import codecs
import time
import random
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np
from math import *
import numpy.ma as ma

def r(val):
    return int(np.random.random() * val)

def random_scale(x,y):
    #对x随机scale,生成x-y之间的一个数
    gray_out = r(y+1-x) + x
    return gray_out


def text_Gengray(bg_gray, line):
    gray_flag = np.random.randint(2)
    if bg_gray < line:
        text_gray = random_scale(bg_gray + line, 255)
    elif bg_gray > (255 - line):
        text_gray = random_scale(0, bg_gray - line)
    else:
        text_gray = gray_flag*random_scale(0, bg_gray - line) + (1 - gray_flag)*random_scale(bg_gray+line, 255)
    return text_gray

def GenCh(f,val, data_shape1, data_shape2, bg_gray, text_gray, text_position):
    img=Image.new("L", (data_shape1,data_shape2),bg_gray)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0),val,text_gray,font=f)
    #draw.text((0, text_position),val.decode('utf-8'),0,font=f)
    A = np.array(img)

    #二值化,确定文字精确的左右边界
    if bg_gray > text_gray:
        ret,bin = cv2.threshold(A,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        ret,bin = cv2.threshold(A,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('A',A)
    #cv2.imshow('bin',bin)


    left = -1
    right = 10000
    for i in range(0,bin.shape[1]):
        if np.sum(bin[:,i]) > 0:
            left = i
            break
    for i in range(bin.shape[1]-1,0,-1):
        if np.sum(bin[:,i]) > 0:
            right = i
            break
    dst  = A[:,left:right+1]
    #cv2.imshow('dst',dst)
    #cv2.waitKey()
    return dst

def tfactor(img):
    img[:,:] = img[:,:]*(0.8+ np.random.random()*0.2)
    return img

def Addblur(img, val):
    blur_kernel = random_scale(2,val)
    #print blur_kernel
    #平滑图像
    img = cv2.blur(img,(blur_kernel,blur_kernel))
    return img

def motionBlur(img,val):
    blur_kernel0 = random_scale(2,val)
    blur_kernel1 = random_scale(2,val)
    anchor = (random_scale(0,blur_kernel0-1),random_scale(0,blur_kernel1-1))
    img = cv2.blur(img,(blur_kernel0,blur_kernel1),anchor=anchor)
    return img

def AddNoiseSingleChannel(single):
    diff = (255-single.max())/3    
    noise = np.random.normal(0,1+r(6),single.shape);
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise;
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def rot(img,angel,shape,max_angel,bg_gray):
    size_o = [shape[1],shape[0]]
    size = (shape[1] + int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs(int(sin((float(angel) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0], [0,size_o[1]], [size_o[0],0], [size_o[0], size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size,borderValue=bg_gray)

    return dst

def rotRandrom(img, factor, size, bg_gray):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size, borderValue=bg_gray)
    return dst

class GenText:
    def __init__(self, ch_size=16,imgHeight=16,imgWidth=64,inter=0,bg_gray=-1,text_gray=-1,text_position=0,offset_left=-1):
        self.ch_size  = ch_size
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.inter = inter
        self.bg_gray = args.bg_gray
        self.text_gray = args.text_gray
        self.text_position = args.text_position
        self.offset_left = args.offset_left

    def draw(self,val,font):
        if self.bg_gray == -1:
            bg_gray = r(256) #随机生成背景灰度
        else: 
            bg_gray = self.bg_gray
        
        #bg_gray = 0
        if self.text_gray == -1:
            text_gray = text_Gengray(bg_gray, 60)#生成前景灰度
        else: 
            text_gray = self.text_gray
        #text_gray = random_scale(30,256)
        if self.text_position == 0:
            text_position = random_scale(0,(self.imgHeight-self.ch_size)/2) #垂直方向文本位置
        else: 
            text_position = self.text_position
        print 'text_pos: ',text_position
        if self.offset_left == -1:
            offset_left = int(np.random.random() * self.ch_size)
        else :
            offset_left = self.offset_left
        offset = offset_left
        ch_num = len(val)
        imgWidth = min(self.imgWidth,offset+ch_num*self.ch_size)
        img = np.array(Image.new("L", (imgWidth, self.imgHeight), bg_gray))
        base = offset_left
        
        #文本间隙
        if self.inter == 0:
            inter = random.randint(1,3)
        else:
            inter = self.inter
        print inter
        writeTxt = ''
        for i in range(ch_num):
            if (base+self.ch_size) <= imgWidth:
                tmp = GenCh(font,val[i], self.ch_size, self.imgHeight, bg_gray, text_gray, text_position)
                img[0: self.imgHeight, base : base + tmp.shape[1]]= tmp
                base += tmp.shape[1]+inter
                writeTxt += val[i]
            else:
                break
        return img, bg_gray,text_gray,writeTxt
    
    def changeBG(self,input,fg_gray,bgImg):
        assert len(bgImg.shape) == 2
        if bgImg.shape[0] < input.shape[0] or bgImg.shape[1] < input.shape[1]:
            return input
        thresh = 40
        if bgImg.shape[0]-input.shape[0]-1 <= 0 or bgImg.shape[1]-input.shape[1]-1<=0:
            return input

        st_y = random.randint(0,bgImg.shape[0]-input.shape[0]-1)
        st_x = random.randint(0,bgImg.shape[1]-input.shape[1]-1)

        tmp = bgImg[st_y:st_y+input.shape[0],st_x:st_x+input.shape[1]]
        
        mean = np.mean(tmp)
        if abs(fg_gray - mean) < thresh:
            return input
        else:
            output = tmp.copy() 
        h = input.shape[0]
        w = input.shape[1]
        ret,input = cv2.threshold(input,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        output[input>0] = fg_gray
        return output

    def generate(self,text,font):
        fg, bg_gray,fg_gray,txt = self.draw(text,font)
        #旋转
        com = rot(fg,r(90)-45,fg.shape,45, bg_gray)

        #更换背景图片
        #com = self.changeBG(com,fg_gray,bgImg)
        #com = rotRandrom(fg,2,(fg.shape[1],fg.shape[0]), bg_gray)
        #com = tfactor(com)
        #2 低通滤波器
        com = motionBlur(com,2)
        com = AddNoiseSingleChannel(com)
        if com.shape[1] < self.imgWidth:
            tmp = np.zeros((self.imgHeight,self.imgWidth),dtype='uint8')
            tmp[:,:] = 128
            tmp[:,0:com.shape[1]] = com.copy()
            com = tmp.copy()
        elif com.shape[1] > self.imgWidth: #rot时，可能会宽度增加一点
            com = cv2.resize(com,(self.imgWidth,self.imgHeight))
        return com,txt


 
  
def genTextImg(args):
        num = args.num
        text_length = args.text_length
        font_size = args.font_size
        font_id = args.font_id
        inter = args.inter
        bg_gray = args.bg_gray
        text_gray = args.text_gray
        text_position = args.text_position
        offset_left = args.offset_left
    
	maxNum = 12
	Gs = []
	fonts = []
    
	if font_size == 0:
    # 随机设置font_size
		font_sizes = [15,16,17,18,19,20] #font_sizes = [26,27,28,29,30]
		for size in font_sizes:
			size = size + inter
			Gs.append(GenText(size,size,(size+1)*(text_length),inter))
			tmp = []
			tmp.append(ImageFont.truetype('./data/fonts/more_font/仿宋_GB2312.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/华文隶书.TTF',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/宋体_GB18030+%26+新宋体_GB18030.ttc',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/微软vista黑体.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/方正楷体GBK.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/方正隶书简体.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/楷体_GB2312.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font//造字工房尚黑G0v1纤细长体.otf',font_size))
			fonts.append(tmp)
	else:
		for step in range(0,text_length,1) :
			font_size=font_size + inter
			Gs.append(GenText(font_size,font_size,(font_size+1)*(text_length-step),inter))
			tmp = []
			tmp.append(ImageFont.truetype('./data/fonts/more_font/仿宋_GB2312.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/华文隶书.TTF',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/宋体_GB18030+%26+新宋体_GB18030.ttc',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/微软vista黑体.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/方正楷体GBK.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/方正隶书简体.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font/楷体_GB2312.ttf',font_size))
			tmp.append(ImageFont.truetype('./data/fonts/more_font//造字工房尚黑G0v1纤细长体.otf',font_size))
			fonts.append(tmp)
	print(len(Gs))
   


	outputPath = 'images'
	txtPath = "data/newsgroup/test/"
        txtFiles = os.listdir(txtPath)
        index=0
        for file in txtFiles:
            fullPath = txtPath + file
            ##输入文档
            with open(fullPath, "rb") as f: 
                    content = f.readlines()
                    f.close()

            #index
            files = os.listdir(outputPath)
            #index = len(files) + 1
            for txt in content:
                    txt = txt.strip()
                    unicode1 = txt.decode('utf-8')
                    if unicode1 == u"\n":
                            continue
                    #flag = random.randint(1,10) <= 9 #写8个字的概率 0.8
                    #if flag:
                    #        count = text_length
                    #else:
                            #count=int(index/1000)+1  #count = random.randint(2,maxNum-1)
                    count = text_length            
                    lines = [unicode1[i:i+count+10] for i in range(0, len(unicode1), count+10)] # 
                    for line in lines:
                            newline = ''
                            punc_count = 0
                            for ch in line:
                                if (ord(ch)>40869 or ord(ch)<19968)and ord(ch)!=12288:
                                    punc_count +=1
                                    print(ch)
                                if True and ord(ch)!=12288:
                                    newline += ch #字符如果在我的字库中，我才生成图像
                                newline=newline.replace(" ","")
                                if len(newline)==count:
                                    break    
                            
                            if len(newline)<count:
                                    continue
                            if punc_count==text_length:
                                break

                            index += 1
                            print index,newline,len(newline), punc_count
                            filename =  str(index) + ".jpg"
                            #writePath =outputPath +'/'+str(len(newline))+'/'+ filename
                            #writePath ='/home/yuz/lijiahui/ocr/background_judge/sentencedata/1/ex_' +filename
                            writePath = '/home/jovyan//SynthText_Chinese/images/'+ filename
                            print writePath
                            Gid = random.randint(0,len(Gs)-1)
                            Gid = punc_count
                            
                            if font_id == 0:
                                fontid = random.randint(0,len(fonts[Gid])-1)
                            else:
                                fontid = font_id
                                
                            img,res_txt = Gs[Gid].generate(newline,fonts[Gid][fontid])
                            cv2.imwrite(writePath,img)
                            if index >= num:
                                break
                    if index >= num:
                        break
        #fin.close()

if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser(description='Genereate Synthetic Text Images')
	parser.add_argument('--num',type=int,dest='num',default=10000,help='flag for the number of images')
	parser.add_argument('--text_length',type=int,dest='text_length',default=0,help='flag for the length of text in the image, default means random????')
	parser.add_argument('--font_size',type=int,dest='font_size',default=0,help='flag for the size of font')
	parser.add_argument('--font_id',type=int,dest='font_id',default=0,help='flag for the type of font,range from 1 to 8, default means random')
	parser.add_argument('--inter',type=int,dest='inter',default=0,help='flag for the inter between text, range from 1 to 3')
	parser.add_argument('--bg_gray',type=int,dest='bg_gray',default=-1,help='flag for the background color, default means random')
	parser.add_argument('--text_gray',type=int,dest='text_gray',default=-1,help='flag for the text color, default means random')
	parser.add_argument('--text_position',type=float,dest='text_position',default=0,help='flag for the text_position, default means random')
	parser.add_argument('--offset_left',type=int,dest='offset_left',default=-1,help='flag for the left offset, default means random')
	args = parser.parse_args()
	genTextImg(args)
	
	




