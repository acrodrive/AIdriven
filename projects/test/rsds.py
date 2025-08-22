import cv2
import socket
import logging
import numpy as np
import torch
import argparse

from ultralytics import YOLO

class RSDS_Camera:
    def __init__(self, ip="host.docker.internal", port=2210, log_level=logging.INFO, name = "CAM00"):
        self.logger = logging.getLogger("RSDS_Camera")
        self.logger.setLevel(log_level)
        self.ip                     = ip
        self.port                   = port
        self.socket                 = None
        self.flag_unsuf_data_len    = 0
        self.flag_first_cycle       = 1
        self.splitdata              = ""
        self.imgtype                = ""
        self.img_size               = ""
        self.data_len               = ""
        self.image_h                 = ""
        self.image_w                = ""
        self.lastdata               = ""
        self.size                   = ""
        self.img                    = np.zeros((640,640,3), np.uint8)
        self.name                   = name

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        self.socket.settimeout(1)
        data = self.socket.recv(64)
        # print(data)

    def data_reception(self):
        self.img = np.zeros((640,640,3), np.uint8)
        try:
            data = self.socket.recv(64)
        except socket.timeout:
            return -1

        if self.flag_first_cycle == 1:
            self.splitdata = data.decode().split(" ")
            self.imgtype = self.splitdata[2]
            self.img_size = self.splitdata[4]
            self.data_len = int(self.splitdata[5])
            self.image_h = int(self.img_size.split('x')[1])
            self.image_w = int(self.img_size.split('x')[0])
            self.lastdata = b''
            self.size = 0
            
            print(data)
            print("imgtype ="   , self.imgtype)
            print("img_size ="  , self.img_size)
            print("data_len ="  , self.data_len)
            print("image_h ="   , self.image_h)
            print("image_w ="   , self.image_w)
            
        self.flag_first_cycle = 0
        self.lastdata = b''
        self.size = 0

        while(self.size < self.data_len):
            try:
                data = self.socket.recv(self.data_len - self.size)
            except socket.timeout:
                continue

            try:
                strdata = data.decode()
                if strdata[0] == '*' and strdata[1] == 'V':
                    self.lastdata = b''
                    self.size = 0
                    continue
            except:
                pass
            
            self.lastdata += data
            self.size = np.frombuffer(self.lastdata, dtype=np.uint8).size
        
        datalist = np.frombuffer(self.lastdata, dtype=np.uint8)
        
        if(self.imgtype == "rgb" and self.size == self.data_len):
            self.img = datalist.reshape((self.image_h, self.image_w, 3))
            
        elif(self.imgtype == "grey16raw" and self.size == self.data_len):
            self.img = datalist.reshape((self.image_h, self.image_w, 2))
            g, b = cv2.split(self.img)
            r = np.zeros_like(g)
            # self.img = cv2.merge((r, g, b))
            self.img = g*4
        
        else:
            self.flag_unsuf_data_len = 1

        return self.img


if __name__ == '__main__':
    
    Sim_State, Sim_Time, Driver_Recon_Speed = 0, 0, 0

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = YOLO('yolo11m.pt')
    model.eval()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add command-line argument(s)
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('-vis', action='store_true', help='active visualization')
    parser.add_argument('-export', choices=['rtsp', 'udp'], default='false', help='specify export mode')
    parser.add_argument('-format', choices=['rawvideo', 'h265'], default='rawvideo', help='specify export format')
    parser.add_argument('-server_ip', default='127.0.0.1', help='current PC ip / 127.0.0.1')
    parser.add_argument('-dst_ip', default='127.0.0.1', help='target PC ip / 127.0.0.1')
    parser.add_argument('-dst_port', default='8554', help='target Program port / 8554')
    parser.add_argument('-dst_endp', default='stream', help='target end point / stream')
    
    # Initialize RSDS Camera
    cam = RSDS_Camera()
    cam.connect()

    # asyncio.run(EnableToOverwriteUAQ())

    # Parse the command-line arguments
    cam.args = parser.parse_args()
    
    try:
        #cap = cv2.VideoCapture(r"C:\Users\ipg\Downloads\ipgkr-rsds_camera_python\example.mp4")

        while 1:
            #ret, frame = cap.read()

            #if not ret : break
            
            cam.data_reception()

            # cam.img = cv2.cvtColor(cam.img, cv2.COLOR_RGB2BGR)
            
            frame = cam.img
            height, width, channels = frame.shape

            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pred = model(frame, conf=0.4, imgsz=(256, 704), device=device)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            boxes = pred[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)   # 클래스 ID (정수형)
            confs = boxes.conf.cpu().numpy()                  # confidence
            xyxy = boxes.xyxy.cpu().numpy()                   # 좌표 [x1, y1, x2, y2]

            names = model.names

            """# 바운딩박스 그리기
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                cls_id = class_ids[i]
                conf = confs[i]
                label = f"{names[cls_id]} {conf:.2f}"

                # 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 라벨 그리기
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

            frame = cv2.resize(frame, (1408, 512))
            cv2.imshow("Recognition Result", frame)"""

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        # Perform cleanup or termination procedures
        print("\n---------------------------\n"
              "Program terminated by user.\n"
              "---------------------------")
    
    # asyncio.run(DeactiveToOverwriteUAQ())
    cam.socket.__exit__()


