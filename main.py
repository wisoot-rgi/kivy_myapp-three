from kivy.app import App
from kivy.uix.label import Label
#from kivy.uix.image import Image

class MyApp(App):
    def build(self):
        #return Label(text="Hello, World!")

        from ultralytics import YOLO
        import cv2
        import math 
        # start webcam
        cap = cv2.VideoCapture('D:/download/interference/kivy_app/images/soda15-6.jpg')
        #cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 640)

        # model
        #model = YOLO("D:/download/interference/yolov8n.pt")# pretrained YOLOv8n model
        model = YOLO('D:/download/interference/kivy_app/images/best.pt')# pretrained YOLOv8n model
        #model = YOLO('C:/Users/Wisoot.K/Downloads/weightszip/weights/best.pt')# pretrained YOLOv8n model

        code_map = {
            "1100100111111" : 1,
            "1100101011111" : 2,
            "1101001011111" : 3,
            "1100110011111" : 4,
            "1101010011111" : 5,
            "1110010011111" : 6,
            "1100101101111" : 7,
            "1101001101111" : 8,
            "1100110101111" : 9,
            "1111110010011" : 10,
            "1111001001111" : 11,
            "1101010101111" : 12,
            "1110010101111" : 13,
            "1101100101111" : 14,
            "1110100101111" : 15,
            "1100111001111" : 16,
            "1101011001111" : 17,
            "1110011001111" : 18,
            "1101010111011" : 19,
            "1111101010011" : 20,
            "1111010101011" : 21,
            "1110101010111" : 22,
            "1101101001111" : 23,
            "1110101001111" : 24,
            "1100101110111" : 25,
            "1101001110111" : 26,
            "1100110110111" : 27,
            "1101010110111" : 28,
            "1101110101011" : 29,
            "1111101001011" : 30,
            "1111010100111" : 31,
            "1111001011011" : 32,
            "1110011100111" : 33,
            "1110010110111" : 34,
            "1101100110111" : 35,
            "1110100110111" : 36,
            "1100111010111" : 37,
            "1101011010111" : 38,
            "1101100111011" : 39,
            "1111100110011" : 40,
            "1111010011011" : 41,
            "1111001010111" : 42,
            "1110110100111" : 43,
            "1101101011011" : 44,
            "1110011010111" : 45,
            "1101101010111" : 46,
            "1101110010111" : 47,
            "1100111100111" : 48,
            "1101110011011" : 49,
            "1111100101011" : 50,
            "1111010010111" : 51,
            "1110111010011" : 52,
            "1110110011011" : 53,
            "1110101100111" : 54,
            "1101011101011" : 55,
            "1101011100111" : 56,
            "1101101100111" : 57,
            "1101110100111" : 58,
            "1100111011011" : 59,
            "1111100100111" : 60,
            "1111001110011" : 61,
            "1110111001011" : 62,
            "1110110010111" : 63,
            "1110101011011" : 64,
            "1110011101011" : 65,
            "1100111110011" : 66,
            "1100101111011" : 67,
            "1101001111011" : 68,
            "1101101110011" : 69,
            "1111011010011" : 70,
            "1111001101011" : 71,
            "1110110110011" : 72,
            "1110101110011" : 73,
            "1110100111011" : 74,
            "1110011011011" : 75,
            "1101111010011" : 76,
            "1101101101011" : 77,
            "1100110111011" : 78,
            "1100111101011" : 79,
            "1111011001011" : 80,
            "1111001100111" : 81,
            "1110110101011" : 82,
            "1110101101011" : 83,
            "1110011110011" : 84,
            "1110010111011" : 85,
            "1101111001011" : 86,
            "1101110110011" : 87,
            "1101011011011" : 88,
            "1101011110011" : 89,
            "1111010110011" : 90
        }

        def get_value_by_key(key):
            return str(code_map.get(key, None))

        while True:
            success, img = cap.read()
            results = model(img, stream=True)

            # coordinates
            result_list=[]
            for r in results:
                boxes = r.boxes
                result_list=r.boxes.xywh.tolist()
                #---------------------------------
            
                temp_data = []
                distance_ratio=[]
                for i in range (len(result_list)):
                    dict_data = {
                    "x" :result_list[i][0],
                    "y" :result_list[i][1], 
                    "w" :result_list[i][2], 
                    "h" :result_list[i][3],       
                    }
                    temp_data.append(dict_data)
                newlist = sorted(temp_data, key=lambda d: d['x'])
                print("new list=",newlist)    

                # //////////////////////////////////////////////
                for i in range(len(newlist)-1):
                    if(newlist[i+1]['w']/newlist[i]['w']>0.80): 
                        ratio = (newlist[i+1]['x']-newlist[i]['x'])/newlist[i]['w']
                    else:
                        ratio = (newlist[i+1]['x']-newlist[i]['x'])/newlist[i+1]['w']
                    
                    if(ratio>=1.0 and ratio<=5.5):
                       distance_ratio.append(ratio)
                    elif (ratio>5.5 and ratio<=6.0):
                       ratio=ratio/2
                       distance_ratio.append(ratio)
                       distance_ratio.append(ratio)
                #///////////////////////////////////////////////
                dot_code = []
                for i in distance_ratio:
                    dot_code.append('1')
                    if   (i>1.9 and i <=3.6) :
                     dot_code.append('0')
                    elif (i>3.6 and i <=5.5):
                     dot_code.append('0')
                     dot_code.append('0')
                dot_code.append('1')    
                print("")
                print("distance_ratio=",distance_ratio)
                print("dot code of","test","=",dot_code)
                print("dot code length=",len(dot_code))
                print("=================================================")   
                
                #////////////////////////////////////////////////////////
                
                if len(dot_code) == 13 :
                #code="invalid_Read"
                 code=get_value_by_key(str(''.join(dot_code)))
                else :
                 code="invalid_Read"
                
                #---------------------------------
            x_position=[]
            y_position=[]    
            for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values       
                    x_position.append(x1)
                    y_position.append(y1)
                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])
                    #print("Class name -->", classNames[cls])

            # object details
            try:
                x_position.sort()
                y_position.sort()
                org = [x_position[0], y_position[0]-50]
                line= [x_position[0], y_position[0]-30]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = (x_position[8]-x_position[0])*2/1000
                color = (0, 0, 255)
                thickness = int((x_position[8]-x_position[0])*3/1000)
                print("x-positio=",x_position)
                print("y-position",y_position)

                cv2.putText(img,"mold number : "+code, org, font, fontScale, color, thickness)
                cv2.putText(img,"..............................", line, font, fontScale, (255,0,255), thickness)

                img=cv2.resize(img, (480,480))
                cv2.imshow('Image_mold number :'+code, img)
                if cv2.waitKey(20000) == ord('q'):break
            
            except:
                img=cv2.resize(img, (320,320))
                cv2.imshow('Webcam_mold number :', img)
                if cv2.waitKey(2) == ord('q'): break
                    
        cap.release()
        cv2.destroyAllWindows()

        def code_position(result_list):
            temp_predict = []
            for i in range (len(result_list)-1) :
                dict_data = {
                "x" :result_list[i][0],
                "y" :result_list[i][1], 
                "w" :result_list[i][2], 
                "h" :result_list[i][3],       
                }
                temp_predict.append(dict_data)
            newlist = sorted(temp_predict, key=lambda d: d['x'])
            return newlist                

                

                



        

if __name__ == "__main__":
    MyApp().run()
