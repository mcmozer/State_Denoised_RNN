% v1 is first run without matched samples
% v2 is second run with matched samples
% then i combined them with analyze_combined and get the final version

% columns: #steps test1 mean / median / std   test2 mean / median /std

sdrnn =[
11 0.950334 0.001855   0.957900 0.001541
17 0.900585 0.003974   0.937450 0.003181
23 0.838794 0.006762   0.921100 0.003810
29 0.788142 0.007162   0.899900 0.004013
];
rnn=[
11 0.938013 0.001718   0.958550 0.001589
17 0.884774 0.004309   0.928250 0.002934
23 0.836330 0.004890   0.905500 0.003337
29 0.786966 0.005955   0.880350 0.003950
];
rnna = [
11 0.930164 0.003016   0.931850 0.002863
17 0.854286 0.005280   0.889800 0.004934
23 0.752474 0.007902   0.834800 0.005906
29 0.677218 0.009036   0.788800 0.006098
];





lw=2;
fs=20;
FS=22;
col=[27 158 119; 217 95 2; 117 112 179]/255;
for i = 1:2
   offset = 2+(i-1)*2;

   subplot(1,4,i)
   hold off
   errorbar(sdrnn(:,1),sdrnn(:,offset),sdrnn(:,offset+1),'linewidth',lw,'color',col(3,:))
   hold on
   set(gca,'fontsize',fs);
   errorbar(rnn(:,1),rnn(:,offset),rnn(:,offset+1),'linewidth',lw, 'color',col(1,:))

   errorbar(rnna(:,1),rnna(:,offset),rnna(:,offset+1),'linewidth',lw,'color',col(2,:))
   if (i==1)
      set(gca,'ylim',[.66 .96])
      title('(a) Majority: Novel Seq.','fontsize',fs);
   else
      set(gca,'ylim',[.77 .96])
      title('(b) Majority: Noisy Seq.','fontsize',fs);
   end
   ylabel('Proportion Correct','fontsize',FS);
   set(gca,'ytick',.65:.05:1.00);
   ytickformat('%,.2f')
   legend('SDRNN','RNN','RNN+A','location','southwest')
   set(gca,'xtick',rnna(:,1))
   xlabel('Sequence Length', 'fontsize',FS);
   axis square

end

reber_sdrnn= [
 50 0.661060 0.002695
100 0.768430 0.003243
200 0.890785 0.003843
400 0.972910 0.002176
800 0.989185 0.001930
];

reber_rnn= [
 50 0.651060 0.002676
100 0.730720 0.003599
200 0.842275 0.004111
400 0.946050 0.002546
800 0.981890 0.002417
];

reber_rnna= [
 50 0.645350 0.002559
100 0.735160 0.003914
200 0.827745 0.005344
400 0.923110 0.003607
800 0.954780 0.004042
];


subplot(1,4,4)
hold off
errorbar(1:5,reber_sdrnn(:,2),reber_sdrnn(:,3),'linewidth',lw,'color',col(3,:))
hold on
set(gca,'fontsize',fs);
errorbar(1:5,reber_rnn(:,2),reber_rnn(:,3),'linewidth',lw, 'color',col(1,:))
errorbar(1:5,reber_rnna(:,2),reber_rnna(:,3),'linewidth',lw,'color',col(2,:))
set(gca,'ylim',[.65 1])
ylabel('Proportion Correct','fontsize',FS);
title('(d) Reber: Novel Seq.','fontsize',fs);
set(gca,'ytick',.65:.05:.95);
set(gca,'ylim',[.64 1])
ytickformat('%,.2f')
legend('SDRNN','RNN','RNN+A','location','southeast')
set(gca,'xlim',[.75 5.25])
xlabel('# Training Examples', 'fontsize',FS);
set(gca,'xticklabel',reber_rnn(:,1))
set(gca,'xtick',1:5)
axis square

subplot(1,4,3)
axis square
title('(c) Reber Grammar','fontsize',fs)
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'box','off')

set(gcf,'position',[10 10 1532 483]);
set(gcf,'paperpositionmode','auto')
print2pdf majority_results.pdf
%system('/Library/TeX/texbin/pdfcrop majority_results.pdf')
%system('cp majority_results.pdf ~/Dropbox/Apps/ShareLaTeX/NIPS2018\ SDRNN/fig');
