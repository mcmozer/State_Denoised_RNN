
import sys
import re
import numpy

NUM_REPLICATIONS = 100

for i in range(1,len(sys.argv)):
   file = open(sys.argv[i],"r");
   x= file.read();
   m = re.search('test1 indiv runs\s*\[(.*?)\].*test2 indiv runs\s*\[(.*?)\]',x);
   m = re.search('test1 indiv runs\s*\[(.*?)\]',x);
   if (m != None):
       test1_string = m.group(1)
       test1_list = test1_string.split(',')
       test1_num = map(float, test1_list)
       test1_num = test1_num[:100] # deal with fact i did 200 replications of rnn
       m = re.search('test2 indiv runs\s*\[(.*?)\]',x);
       test2_string = m.group(1)
       test2_list = test2_string.split(',')
       test2_num = map(float, test2_list)
       test2_num = test2_num[:100] # deal with fact i did 200 replications of rnn
       print("%-25s %.4f %.4f %.4f   %.4f %.4f %.4f" % (sys.argv[i], 
                 numpy.mean(test1_num), numpy.median(test1_num), 
                 numpy.std(test1_num) / numpy.sqrt(float(len(test1_num))),
                 numpy.mean(test2_num), numpy.median(test2_num), 
                 numpy.std(test2_num) / numpy.sqrt(float(len(test2_num)))))
   file.close()
