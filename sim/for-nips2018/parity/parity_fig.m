FS=18*20/18; % font size of axis labels
fs = 14*20/18;
lw = 2;
cs = 40;
col=[27 158 119; 217 95 2; 117 112 179]/255;
allresults
clf


%                      test1 mean+SEM       test2 mean+SEM
%  corrected for between subject variability
rnn_tanh=[            0.566380 0.014075   0.729180 0.006699];
sdprnn_tanh=[         0.795013 0.015146   0.911074 0.005810];
rnna_tanh=[           0.496680 0.010554   0.620020 0.006282];
g(1,:,:) = [rnn_tanh; sdprnn_tanh; rnna_tanh];

rnn_GRU=[             0.637786 0.014895   0.777129 0.007542];
sdprnn_GRU=[          0.898620 0.012740   0.937891 0.006785];
rnna_GRU=[            0.603776 0.013841   0.709141 0.008672];
g(2,:,:) = [rnn_GRU; sdprnn_GRU; rnna_GRU];

glabels = {'tanh','GRU'};

alabels={'a','c','e','g','b','d','f','h'};
[ha, pos] = tight_subplot(2,4,[.15 .04],[.15 .07], [.07 .07]);
for test=1:2 % which test data set

   for grap = 1:2  % tanh and GRU
      offset = 1+(test-1)*2;
      aa=1+4*(test-1)+grap-1;
      axes(ha(aa))

      hold off
      bar(1,g(grap,1,offset),'facecolor',col(1,:), 'edgecolor',col(1,:))
      hold on
      q=errorbar(1,g(grap,1,offset),g(grap,1,1+offset), 'linewidth', lw, 'color',col(1,:)*.7);
      q.CapSize = cs;
      set(gca,'fontsize',fs)

      bar(3,g(grap,2,offset),'facecolor',col(3,:), 'edgecolor',col(3,:))
      q=errorbar(3,g(grap,2,offset),g(grap,2,1+offset), 'linewidth', lw, 'color',col(3,:)*.7);
      q.CapSize = cs;

      bar(2,g(grap,3,offset),'facecolor',col(2,:), 'edgecolor',col(2,:))
      q=errorbar(2,g(grap,3,offset),g(grap,3,1+offset), 'linewidth', lw, 'color',col(2,:)*.7);
      q.CapSize = cs;
      set(gca,'xtick',1:3)
      set(gca,'xlim',[.5 3.5])
      l1 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}RNN',col(1,:));
      l2 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}RNN+A', col(2,:));
      l3 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}SDRNN', col(3,:));
      set(gca,'xticklabel',{l1 l2 l3})
      xlabel(['Architecture (' glabels{grap} ')'],'fontsize',FS);
      ytickformat('%.1f')
      set(gca,'ylim',[.4 1.0])
      yl = get(gca,'ylim');
      l = floor(yl*10)/10;
      set(gca,'ytick',l(1):.1:l(2))
     
      if (test == 1 && grap==1)
	      ylabel({'New Sequences' 'Proportion Correct'},'fontsize',FS)
      elseif (test==2 && grap==1)
	      ylabel({'Additive Noise' 'Proportion Correct'},'fontsize',FS)
      end
      
      xl = get(gca,'XLim'); yl = get(gca,'YLim');
      text(xl(1)+.015*(xl(2)-xl(1)),yl(2)-.07*(yl(2)-yl(1)),['(' alabels{aa} ')'],'fontweight','bold','fontsize',FS)
      set(gca,'fontsize',fs)
   end

   datan = [125         0.712057 0.016816   0.872949 0.007077
            250         0.727826 0.016889   0.889375 0.006061
            375         0.777773 0.016365   0.902129 0.006681
            500         0.795013 0.018246   0.911074 0.007013
            625         0.804570 0.016442   0.899316 0.007687
            750         0.796589 0.015168   0.882754 0.007015
            ];

   % varying noise level
   aa=3+4*(test-1);
   axes(ha(aa))
   %subplot(2,4,3+4*(test-1))
   offset = 2+(test-1)*2;
   errorbar(1:size(datan,1),datan(:,offset),datan(:,offset+1),'linewidth',lw,'color',col(3,:))
   set(gca,'xticklabel',{'.125' '.250' '.375' '.500' '.625' '.750'})
   xlabel('Noise Standard Deviation','fontsize',FS);
   set(gca,'xtick',1:size(datan,1))
   ytickformat('%.2f')
   %l = floor(get(gca,'ylim')*10)/10;
   set(gca,'fontsize',fs)
   if (test==1)
      set(gca,'ytick',.6:.05:1)
      set(gca,'ylim',[.675 .85])
   else
      set(gca,'ytick',.6:.05:1)
      set(gca,'ylim',[.80 .95])
   end
   set(gca,'xlim',[.7 size(datan,1)+.3])
         
   xl = get(gca,'XLim'); yl = get(gca,'YLim');
   text(xl(1)+.015*(xl(2)-xl(1)),yl(2)-.07*(yl(2)-yl(1)),['(' alabels{aa} ')'],'fontweight','bold','fontsize',FS)
   
   % varying weight decay
   datawd = [00000       0.795013 0.015799   0.911074 0.006593
             00625       0.795378 0.015582   0.905625 0.006881
             01250       0.760586 0.014095   0.895391 0.006424
             02500       0.804596 0.014275   0.907246 0.005847
             05000       0.804805 0.014841   0.907988 0.006521
             10000       0.780573 0.014854   0.900938 0.006766
             20000       0.818971 0.015832   0.920234 0.006882
             ];

          
   aa=4+4*(test-1);
   axes(ha(aa))
   hold off
   errorbar(1:size(datawd,1),datawd(:,offset),datawd(:,offset+1),'linewidth',lw,'color',col(3,:))
   hold on
   %subplot(2,4,4+4*(test-1))
   set(gca,'xticklabel',{'-\infty' '-4' '-3' '-2' '-1' '0' '1' '2'})
   set(gca,'xticklabel',{'0^{ }','2^{-4}','2^{-3}','2^{-2}','2^{-1}','2^{0}','2^{1}','2^{2}'})
   set(gca,'xtick',1:size(datawd,1))
   xlabel('Weight Decay','fontsize',FS)
   ytickformat('%.2f')
   %l = floor(get(gca,'ylim')*10)/10
   set(gca,'xlim',[.5 size(datawd,1)+.5])
   set(gca,'fontsize',fs)
   if (test==1)
      set(gca,'ytick',.70:.05:.85)
      set(gca,'ylim',[.675 .85]);
   else
      set(gca,'ytick',.6:.05:1)
      set(gca,'ylim',[.80 .95]);
   end
        
   xl = get(gca,'XLim'); yl = get(gca,'YLim');
   text(xl(1)+.015*(xl(2)-xl(1)),yl(2)-.07*(yl(2)-yl(1)),['(' alabels{aa} ')'],'fontweight','bold','fontsize',FS)
   
end
set(gcf,'position',[100 100 1243 420])
print2pdf parity_fig.pdf
system('cp parity_fig.pdf ~/projects/SDRNN/papers/nips2018/fig');
