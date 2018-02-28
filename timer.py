import time
import argparse

parser = argparse.ArgumentParser()                        
parser.add_argument('--time', type=str, dest='sleep_time',
                        default=3600, help='sleep time (unit: hour).')                    
args = parser.parse_args() 

print('Set python to sleep for ' + args.sleep_time + ' hour.')
time.sleep(int(3600*float(args.sleep_time)))
