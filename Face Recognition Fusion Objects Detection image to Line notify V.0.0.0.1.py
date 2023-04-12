# ใช้ pip install opencv-python==4.6.0.66

import cv2
import numpy as np
import face_recognition as face
import numpy as np
from line_notify import LineNotify
import datetime
import pandas as pd
import winsound

frequency = 3500  # Set Frequency To 2500 Hertz
duration = 250  # Set Duration To 1000 ms == 1 second

clock = datetime.datetime.now()
img_counter = 0
number = 0

ACCESS_TOKEN = "tkCTwBNXqG440QZMgXQTVoZEt2s5HezZ96WTLpsDWjY"
notify = LineNotify(ACCESS_TOKEN)


CLASSES = [
    "BACKGROUND",
    "AEROPLANE",
    "BICYCLE",
    "BIRD",
    "BOAT",
    "BOTTLE",
    "BUS",
    "CAR",
    "CAT",
    "CHAIR",
    "COW",
    "DININGTABLE",
    "DOG",
    "HORSE",
    "MOTORBIKE",
    "HUMAN",
    "POTTEDPLANT",
    "SHEEP",
    "SOFA",
    "TRAIN",
    "TVMONITOR"
    ]
CLASSES_NAME = [
    "วัตถุบางสิ่ง",
    "เครื่องบิน", #AIRPLANE
    "จักรยานยนต์",
    "นก",
    "เรือ",
    "ขวด",
    "รถบรรทุก",  #BUS
    "รถยนต์",
    "แมว",
    "เก้าอี้",
    "โค-กระบือ",
    "โต๊ะ",
    "สุนัข",
    "ม้า",
    "จักรยานยนต์",
    "คน",
    "กระถางต้นไม้",
    "แกะ",
    "โซฟา",
    "รถไฟ",
    "จอมอนิเตอร์"
    ]
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
#โหลดmodelจากแฟ้ม
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")





#ดึงวิดีโอตัวอย่างเข้ามา, ถ้าต้องการใช้webcamให้ใส่เป็น0
video_capture = cv2.VideoCapture(0) 

cv2.namedWindow('My Window',cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('My Window',cv2.WND_PROP_ASPECT_RATIO,cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('My Window',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


# ซ้าย บน
angle1 = np.array([
    [(10,10),(50,10)],
    [(10,10),(10,60)]   
                 ])
# ซ้าย ล่าง
angle2 = np.array([
    [(10,425),(50,425)],
    [(10,370),(10,425)]
                  ])
# ขวา ล่าง
angle3 = np.array([
    [(630,425),(630,370)], #   
    [(630,425),(590,425)]  # 
                 ])
# ขวา บน
angle4 = np.array([
    [(590,10),(630,10)], #   
    [(630,10),(630,60)]  # 
                 ])  


# ใบหน้าคนที่ต้องการรู้จำเป็น reference

 

# ร.ต.ธิติ จันทรินทร์  
B26_1_THITI_image = face.load_image_file("People/B26_1_THITI.jpg")
B26_1_THITI_face_encoding = face.face_encodings(B26_1_THITI_image)[0] # B26_1_CHAIYAKON

# No.007
C68_1_Somporn_image = face.load_image_file("People/C68_1_Somporn.jpg")
C68_1_Somporn_face_encoding = face.face_encodings(C68_1_Somporn_image)[0]




# ประกาศตัวแปร
face_locations = []
face_encodings = []
face_names = []
face_percent = []



# ตัวแปรนี้ใช้สำหรับคิดเฟรม/เฟรม เพื่อเพิ่ม fps
process_this_frame = True

known_face_encodings = [
    
    B26_1_THITI_face_encoding, # ร.ต.ธิติ จันทรินทร์ 

    C68_1_Somporn_face_encoding
    
]
known_face_names = [
    
   
    "Thiti", # ร.ต.ธิติ จันทรินทร์ 

    "Somporn"
]
known_face_names0 = [
    
    "THITI JANTHARIN", # ร.ต.ธิติ จันทรินทร์ 

    "SOMPORN TONGSUP"
]
# ชื่อคน > แจ้งลง LINE Notify
known_face_names1 = [
    
    "ร.ต.ธิติ จันทรินทร์", # ร.ต.ธิติ จันทรินทร์

    "พ.อ.อ.สมพร  ทองทรัพย์"
]
# ตำแหน่ง > แจ้งลง LINE Notify
known_face_names2 = [

    "ผบ.มว.ปตอ.ร้อย.ตอ.พัน.อย.รร.การบิน",# ร.ต.ธิติ จันทรินทร์           

    "ผบ.หมู่ ๑ มว.๑ ร้อย.รก.ฯ"
]
# หน่วยงาน > แจ้งลง LINE Notify
known_face_names3 = [
    
    
    "พัน.อย.รร.การบิน", # ร.ต.ธิติ จันทรินทร์                

    "พัน.อย.รร.การบิน"
]

#loopคำนวณแต่ละเฟรมของวิดีโอ
while True:
    #อ่านค่าแต่ละเฟรมจากวิดีโอ
    ret, frame = video_capture.read()
    cv2.rectangle(frame, (0, 435), (1500, 730), (96, 96, 96), -1) # 
    #cv2.putText(frame, str(clock.now()), (380, 470), cv2. FONT_HERSHEY_DUPLEX , 0.7,  # ขนาดฟอนต์   (กำลังดี)
    cv2.putText(frame,"The Falcon Inspiron Team.", (453, 452), cv2. FONT_HERSHEY_COMPLEX_SMALL , 0.55,
        (255, 255, 255), 1) 
    cv2.putText(frame, str(clock.now()), (454, 472), cv2. FONT_HERSHEY_COMPLEX_SMALL, 0.75,  # ขนาดฟอน ต์            
        (255, 255, 0), 1)  # สี + ความหนาอักษร
        #แสดงผลลัพท์ Video



    if ret:

               
        (h,w) = frame.shape[:2]
        # ท ำ preprocessing
       
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
       
        net.setInput(blob)
        #feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
       
        detections = net.forward()



        #ลดขนาดสองเท่าเพื่อเพิ่มfps 
        small_frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
        #เปลี่ยน bgrเป็น rgb 
        rgb_small_frame = small_frame[:,:,::-1]

        face_names = []
        face_percent = []

        if process_this_frame:
            #ค้นหาตำแหน่งใบหน้าในเฟรม 
            face_locations = face.face_locations(rgb_small_frame, )
            #นำใบหน้ามาหาfeaturesต่างๆที่เป็นเอกลักษณ์ 
            face_encodings = face.face_encodings(rgb_small_frame, face_locations)
            
            #เทียบแต่ละใบหน้า
            for face_encoding in face_encodings:
                face_distances = face.face_distance(known_face_encodings, face_encoding)
                best = np.argmin(face_distances)
                face_percent_value = 1-face_distances[best]

                name = known_face_names[best]
                Notify0 = known_face_names0[best] 
                Notify1 = known_face_names1[best] 
                Notify2 = known_face_names2[best]
                Notify3 = known_face_names3[best]

                #กรองใบหน้าที่ความมั่นใจ50% ปล.สามารถลองเปลี่ยนได้  
                if face_percent_value >= 0.55:          
                
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                    face_names.append(name)
                    winsound.Beep(frequency, duration)

        #วาดกล่องและtextเมื่อแสดงผลออกมาออกมา
        for (top,right,bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            top*= 2
            right*= 2
            bottom*= 2
            left*= 2

            color = [0,255,0]

            cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
            cv2.rectangle(frame, (left-1, top -30), (right+1,top), color, cv2.FILLED)
            cv2.rectangle(frame, (left-1, bottom), (right+1,bottom+30), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+6, top-6), font, 0.6, (0,0,0), 1) #  (0,0,0) สีดำ
            #cv2.putText(frame, "MATCH: "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, str(percent)+"%", (left+6, bottom+23), font, 0.6, (0,0,0), 1) #  (0,0,0) สีดำ

        for (top, right, bottom, left), name, percent in zip(face_locations, face_names, face_percent):
            if face_percent_value >= 0.55:
               cv2.circle(frame, (15,460), 10, (0,0,255), -1)
               cv2.putText(frame, Notify0,(28,468), font, 0.8,(0,255,255),2) 

               #cv2.polylines(frame,angle1, angle2, angle3, angle4, 1,(0,0,255),3)
               cv2.polylines(frame,angle1,1,(0,0,255),3)
               cv2.polylines(frame,angle2,0,(0,0,255),3)
               cv2.polylines(frame,angle3,0,(0,0,255),3)
               cv2.polylines(frame,angle4,0,(0,0,255),3)

# บันทึกภาพเข้าโฟลเดอร์                            
               img_name_1 = "PICTURE_1/image_{}.jpg".format(img_counter)
               Picture_1 = cv2.resize(frame , (0,0), fx=0.6,fy=0.6) # fx=0.5,fy=0.5
               cv2.imwrite(img_name_1, Picture_1) # small_frame
               #print("{} written!".format(img_name_1))
               img_counter += 1
               #number += 1
               #print("{} written!".format(img_name_1))

               #number_name = "ลำดับที่ : {} ".format(img_counter)
               number_name = "{}".format(img_counter)
               A = ("{} written!".format(img_name_1))
               B = number_name 
               print("ตรวจใบหน้าที่ : " + B)
               

# ส่ง LINE
               Pic_Notify_1 = img_name_1   # format(img_name_1)
               notify.send("ลำดับที่ : "+ B + "\nชื่อ: " + Notify1 + "\nตำแหน่ง:  "+ Notify2 + "\nหน่วยงาน: "+ Notify3 +"\nความถูกต้อง : "+str(percent)+" %"+"\n ⬇️ ", Pic_Notify_1 )
               #notify.send("ตรวจพบใบหน้า \nชื่อ: " + Notify1 + "\nตำแหน่ง:  "+ Notify2 + "\nหน่วยงาน: "+ Notify3 +"\nความถูกต้อง : "+str(percent)+" %"+"\n  ", )


            else:  #  เมื่อระบบไม่รู้จักใบหน้า   angle
               #cv2.polylines(frame,angle1, angle2, angle3, angle4, 1,(0,0,255),3)
               cv2.polylines(frame,angle1,1,(0,0,255),3)
               cv2.polylines(frame,angle2,0,(0,0,255),3)
               cv2.polylines(frame,angle3,0,(0,0,255),3)
               cv2.polylines(frame,angle4,0,(0,0,255),3)

        for i in np.arange(0, detections.shape[2]):
           
            percent = detections[0,0,i,2]
            #กรองเอาเฉพาะค่าpercentที่สูงกว่า 0.5 เพิ่มลดได้ตามต้องการ
           
            if percent > 0.9900:
            #if percent > 0.50:
               
                class_index = int(detections[0,0,i,1])
               
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
               
                (startX, startY, endX, endY) = box.astype("int")


                #ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ
               
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)


                TEXT_Notify = "{} ที่หน้าบ้าน".format(CLASSES_NAME[class_index], percent*100)
               
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
               
                cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
               
                y = startY - 15 if startY-15>15 else startY+15
               
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
 
                img_name = "PICTURE_1/image_1.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                #img_counter += 1
 
                notifying = img_name        # format(img_name)
                Notify = (" " +  TEXT_Notify)
                ACCESS_TOKEN = "IUmNdJYFQuetWIZDUlbvOBS5vSjXZvJOAwlT8Zcmd03"
                notify = LineNotify(ACCESS_TOKEN)
                 # ส่งข้อความ + ภาพที่อยู่ในโฟลเดอร์เดียวกันนี้
                notify.send(Notify, notifying)
 
                winsound.Beep(frequency, duration)


    cv2.imshow('My Window',frame)
    if cv2.waitKey(1) == ord('q'): 
        break

video_capture .release()
cv2.destroyAllWindows()
 
