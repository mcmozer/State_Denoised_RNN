# this script combines the results from run1 (unmatched samples)
# and run2 (matched samples), generates statistics, and generates a file for
# running an anova
import sys
import re
import numpy
import glob


anova_test1_file = open('anova_test1.in','w');
anova_test2_file = open('anova_test2.in','w');
head = ['rnn','rnna','sdprnn']
for h in head:
    seqlen = ['10','15','20','25','30']
    for s in seqlen:
        path = 'run[12]/' + h + '_' + s + '*.out'
        test1_num = []
        test2_num = []
        for filename in glob.glob(path):
            with open(filename, 'r') as file:
                x= file.read();
                m = re.search('test1 indiv runs\s*\[(.*?)\].*test2 indiv runs\s*\[(.*?)\]',x);
                m = re.search('test1 indiv runs\s*\[(.*?)\]',x);
                if (m != None):
                    test1_string = m.group(1)
                    test1_list = test1_string.split(',')
                    test1_num = test1_num + map(float, test1_list)

                    m = re.search('test2 indiv runs\s*\[(.*?)\]',x);
                    test2_string = m.group(1)
                    test2_list = test2_string.split(',')
                    test2_num = test2_num + map(float, test2_list)
                file.close()
        for i in range(len(test1_num)):
            anova_test1_file.write("%3d %-5s %2s %f\n" % (i,h,s,test1_num[i]))
        for i in range(len(test2_num)):
            anova_test2_file.write("%3d %-5s %2s %f\n" % (i,h,s,test2_num[i]))

        print("%-10s %-2s %.4f %.4f %.4f   %.4f %.4f %.4f" % (h,s, 
            numpy.mean(test1_num), numpy.median(test1_num), 
            numpy.std(test1_num) / numpy.sqrt(float(len(test1_num))),
            numpy.mean(test2_num), numpy.median(test2_num), 
            numpy.std(test2_num) / numpy.sqrt(float(len(test2_num)))))
anova_test1_file.close()
anova_test2_file.close()

