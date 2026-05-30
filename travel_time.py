import math
 
def flow_to_speed(flow):
 
    if flow <= 351:
        return 60
 
    a = -1.4648375
    b = 93.75
    c = -flow
 
    discriminant = b**2 - 4*a*c
 
    speed1 = (-b + math.sqrt(discriminant)) / (2*a)
    speed2 = (-b - math.sqrt(discriminant)) / (2*a)
 
    return max(speed1, speed2)
 
def calculate_travel_time(flow, distance=5):
 
    speed = flow_to_speed(flow)
 
    travel_time = (distance / speed) * 60
 
    return round(travel_time, 2)