
import sys
import re
import numpy

for i in range(1,len(sys.argv)):
   file = open(sys.argv[i],"r");
   fn = re.search('sc_(.*).out$',sys.argv[i])
   fn = re.sub('\.','p',fn.group(1))

   x= file.read();
   m = re.search('test1 indiv runs\s*\[(.*?)\].*test2 indiv runs\s*\[(.*?)\]',x);
   m = re.search('test1 indiv runs\s*\[(.*?)\]',x);
   test1_string = m.group(1)
   print("t_%s(1,:) = [%s];" % (fn, test1_string))

   m = re.search('test2 indiv runs\s*\[(.*?)\]',x);
   test2_string = m.group(1)
   print("t_%s(2,:) = [%s];" % (fn, test2_string))

   file.close()
