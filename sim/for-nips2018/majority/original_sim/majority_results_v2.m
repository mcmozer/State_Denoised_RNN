% columns: #steps test1 mean / median / std   test2 mean / median /std

rnn = [
10 0.9523 0.9540 0.0029   0.9636 0.9650 0.0017
15 0.9010 0.9044 0.0046   0.9400 0.9450 0.0023
20 0.8726 0.8876 0.0075   0.9225 0.9275 0.0042
25 0.8362 0.8520 0.0079   0.9029 0.9100 0.0042
30 0.7910 0.8310 0.0113   0.8820 0.8950 0.0054
];

sdrnn = [
10 0.9637 0.9665 0.0023   0.9633 0.9650 0.0016
15 0.9006 0.9197 0.0079   0.9394 0.9450 0.0025
20 0.8854 0.9005 0.0084   0.9334 0.9400 0.0037
25 0.8388 0.8635 0.0093   0.9225 0.9300 0.0039
30 0.7846 0.8346 0.0127   0.8980 0.9100 0.0049
];

rnna=[
10 0.9554 0.9605 0.0030   0.9428 0.9450 0.0028
15 0.8706 0.8971 0.0095   0.9028 0.9300 0.0075
20 0.7935 0.8251 0.0115   0.8527 0.8650 0.0094
25 0.7354 0.7586 0.0129   0.8208 0.8250 0.0097
30 0.6816 0.6640 0.0120   0.7767 0.7625 0.0094
];

lw=2;
fs=20;
FS=22;
col=[27 158 119; 217 95 2; 117 112 179]/255;
for i = 1:2
   offset = 2+(i-1)*3;

   subplot(1,2,i)
   hold off
   errorbar(sdrnn(:,1),sdrnn(:,offset),sdrnn(:,offset+2),'linewidth',lw,'color',col(3,:))
   hold on
   set(gca,'fontsize',fs);
   errorbar(rnn(:,1),rnn(:,offset),rnn(:,offset+2),'linewidth',lw, 'color',col(1,:))

   errorbar(rnna(:,1),rnna(:,offset),rnna(:,offset+2),'linewidth',lw,'color',col(2,:))
   if (i==1)
      set(gca,'ylim',[.66 .97])
      ylabel('Proportion Correct','fontsize',FS);
      title('Novel Sequences','fontsize',fs);
   else
      set(gca,'ylim',[.77 .97])
      title('Noisy Sequences','fontsize',fs);
   end
   set(gca,'ytick',.65:.05:1.00);
   ytickformat('%,.2f')
   legend('SDRNN','RNN','RNN+A','location','southwest')
   set(gca,'xtick',rnna(:,1))
   xlabel('Sequence Length', 'fontsize',FS);

end
set(gcf,'position',[10 10 645 300]);
print2pdf majority_results_v2.pdf
%system('/Library/TeX/texbin/pdfcrop majority_results.pdf')
%system('cp majority_results.pdf ~/Dropbox/Apps/ShareLaTeX/NIPS2018\ SDRNN/fig');

