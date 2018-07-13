FS=18*20/18; % font size of axis labels
fs = 14*20/18;
allresults
clf
[ha, pos] = tight_subplot(2,4,[.1 .05],[0 0], [0 0]);
for test=1:2
   axes(ha(1+4*(test-1)))
   %subplot(2,4,1+4*(test-1))
   ds = {t_baseline t_500_15_p0000 t_prediction};
   dat = [];
   for j=1:length(ds)
      dat(:,j) = ds{j}(test,:)';
   end
   boxplot(dat,1)
   set(gca,'xticklabel',{'RNN' 'SDRNN' 'RNN+A'})
   xlabel('Architecture','fontsize',FS);
   ytickformat('%.1f')
   yl = get(gca,'ylim');
   l = floor(yl*10)/10;
   set(gca,'ytick',l(1):.1:l(2))
   if (test == 1)
      ylabel({'New Sequences' 'Proportion Correct'},'fontsize',FS)
      text(-.6,1.01,'(a)','fontweight','bold','fontsize',fs)
   else
      ylabel({'Additive Noise' 'Proportion Correct'},'fontsize',FS)
      text(-.6,1.01,'(b)','fontweight','bold','fontsize',fs)
   end
   set(gca,'fontsize',fs)
   
   % varying noise level
   axes(ha(2+4*(test-1)))
   %subplot(2,4,2+4*(test-1))
   ds = {t_125_15_p0000 t_250_15_p0000 t_375_15_p0000 t_500_15_p0000 t_625_15_p0000 t_750_15_p0000};
   dat = [];
   for j=1:length(ds)
      dat(:,j) = ds{j}(test,:)';
   end
   boxplot(dat,1)
   set(gca,'xticklabel',{'.125' '.250' '.375' '.500' '.625' '.750'})
   xlabel('Noise Standard Deviation','fontsize',FS);
   ytickformat('%.1f')
   %l = floor(get(gca,'ylim')*10)/10;
   set(gca,'ytick',l(1):.1:l(2))
   set(gca,'ylim',yl);
   set(gca,'fontsize',fs)
   if (test==1)
      text(-.6,1.01,'(c)','fontweight','bold','fontsize',fs)
   else
      text(-.6,1.01,'(d)','fontweight','bold','fontsize',fs)
   end
   
   % varying #steps
   axes(ha(3+4*(test-1)))
   %subplot(2,4,3+4*(test-1))
   ds = {t_500_2_p0000 t_500_5_p0000 t_500_10_p0000 t_500_15_p0000};
   dat = [];
   for j=1:length(ds)
      dat(:,j) = ds{j}(test,:)';
   end
   boxplot(dat,1)  
   set(gca,'xticklabel',{'2' '5' '10' '15'});
   xlabel('# Attractor Iterations','fontsize',FS)
   ytickformat('%.1f')
   %l = floor(get(gca,'ylim')*10)/10;
   set(gca,'ytick',l(1):.1:l(2))
   set(gca,'ylim',yl);
   set(gca,'fontsize',fs)
   
   if (test==1)
      text(-.21,1.01,'(e)','fontweight','bold','fontsize',fs)
   else
      text(-.21,1.01,'(f)','fontweight','bold','fontsize',fs)
   end
  
   % varying weight decay
   axes(ha(4+4*(test-1)))
   %subplot(2,4,4+4*(test-1))
   ds = {t_500_15_p0000 t_500_15_p0625 t_500_15_p1250 t_500_15_p2500 t_500_15_p5000 t_500_15_1p000 t_500_15_2p000};
   dat = [];
   for j=1:length(ds)
      dat(:,j) = ds{j}(test,:)';
   end
   boxplot(dat,1)  
   set(gca,'xticklabel',{'0.00' '0.06' '0.13' '0.25' '0.50' '1.00' '2.00'})
   xlabel('Weight Decay','fontsize',FS)
   ytickformat('%.1f')
   %l = floor(get(gca,'ylim')*10)/10;
   set(gca,'ytick',l(1):.1:l(2))
   set(gca,'ylim',yl);
   set(gca,'fontsize',fs)
   if (test==1)
      text(-.76,1.01,'(g)','fontweight','bold','fontsize',fs)
   else
      text(-.76,1.01,'(h)','fontweight','bold','fontsize',fs)
   end
end
set(gcf,'position',[10 10 1243 420])
print2pdf parity_fig.pdf
system('cp parity_fig.pdf ~/projects/CognitivelyInformedAI/rnn/papers/nips2018/fig');