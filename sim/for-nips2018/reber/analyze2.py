# this script corrects error bars to be within subject
import sys
import re
import numpy as np

NUM_REPLICATIONS = 100

n_train = [50,100,200,400,800]
condition = ['sdprnn','rnn','rnna'];
save1 = np.zeros((NUM_REPLICATIONS,len(condition),len(n_train)))
save2 = np.zeros((NUM_REPLICATIONS,len(condition),len(n_train)))
for lix in range(len(n_train)):
    for cix in range(len(condition)):
        file = open('results/' + condition[cix]+'_'+str(n_train[lix])+'.out','r')

        x= file.read();
        m = re.search('test1 indiv runs\s*\[(.*?)\].*test2 indiv runs\s*\[(.*?)\]',x);
        m = re.search('test1 indiv runs\s*\[(.*?)\]',x);
        if (m != None):
            test1_string = m.group(1)
            test1_list = test1_string.split(',')
            test1_num = map(float, test1_list)
            save1[:,cix,lix] = test1_num[:NUM_REPLICATIONS]

        file.close()

d1 = np.mean(save1,axis=1,keepdims=True)
d2 = np.mean(d1,axis=0,keepdims=True)
adjust = d2-d1
test1_sem = np.std(save1+adjust,axis=0)/np.sqrt(float(save1.shape[0]))
test1_mean = np.mean(save1,axis=0)

for cix in range(len(condition)):
    print(condition[cix]) + "= ["
    for lix in range(len(n_train)):
        print('%3d %.6f %.6f' % (n_train[lix],test1_mean[cix,lix],test1_sem[cix,lix]))
    print ("];\n")


# MAKE WITHIN SUBJECT ERROR BARS

#data: array with dimensions #subjects X #conditions X #weeks
#(conditions here are the review conditions, weeks are the blocks of material)
#d1 = mean(data, 2); % # subjects X 1 X #weeks
#d2 = mean(d1, 1); % 1 X 1 X #weeks
#adjust = repmat(d2,size(data,1),size(data,2),1) - repmat(d1,1,size(data,2),1);
#dataSEM = squeeze(std(data+adjust,[],1)) ./ sqrt(size(data,1));
#I verified that the mean doesn't change:
#find(squeeze(mean(data+adjust,1)-mean(data,1)) > 1e-10) should return empty matrix
#and i verified that the SEM goes to zero if i define data such that
#data(:,2,:) = data(:,1,:)+constant
