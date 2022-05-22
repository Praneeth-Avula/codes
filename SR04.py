#The code is about measuring the distance from the robot and the obstacle when the robot is moving forward
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor
from time import sleep
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
duty_cycle = 0
LED = 7
Taster = 5
#Linesensors
cny70_left = 16
cny70_right = 26
#SR04
trigger=25
echo=27
Motor1_PWM = 18
Motor1_IN1 = 17
Motor1_IN2 = 22
Motor2_PWM = 19
Motor2_IN1 = 24
Motor2_IN2 = 4
GPIO.setup(Taster, GPIO.IN, pull_up_down = GPIO.PUD_UP)
def M1_forward():
 GPIO.output(Motor1_IN2,GPIO.LOW)
 GPIO.output(Motor1_IN1,GPIO.HIGH)
def M1_backward():
 GPIO.output(Motor1_IN1,GPIO.LOW)
 GPIO.output(Motor1_IN2,GPIO.HIGH)
def M2_forward():
 GPIO.output(Motor2_IN2,GPIO.LOW)
 GPIO.output(Motor2_IN1,GPIO.HIGH)

def M2_backward():
 GPIO.output(Motor2_IN1,GPIO.LOW)
 GPIO.output(Motor2_IN2,GPIO.HIGH)
GPIO.setup(Motor1_IN1,GPIO.OUT)
GPIO.setup(Motor1_IN2,GPIO.OUT)
GPIO.setup(Motor1_PWM,GPIO.OUT)
PWM_1 = GPIO.PWM(Motor1_PWM, 90) #GPIO als PWM mit Frequenz 90Hz
PWM_1.start(0) #Duty Cycle = 0
GPIO.setup(Motor2_IN1,GPIO.OUT)
GPIO.setup(Motor2_IN2,GPIO.OUT)
GPIO.setup(Motor2_PWM,GPIO.OUT)
PWM_2 = GPIO.PWM(Motor2_PWM, 90) #GPIO als PWM mit Frequenz 90Hz
PWM_2.start(0) #Duty Cycle = 0
i = 0
GPIO.setup(LED, GPIO.OUT)
sensor = DistanceSensor(echo=27, trigger=25)
while i<6:
 PWM_1.ChangeDutyCycle(40)
 M1_forward()
 PWM_2.ChangeDutyCycle(40)
 M2_forward()
 print("Distance: ", sensor.distance * 100)
 sleep(1)
 i=i+1
