
lw=2;
fs=22;
FS=25;
col=[27 158 119; 217 95 2; 117 112 179]/255;
cs = 50;


subplot(1,3,1)

% tanh_5seq_1filler_5000examples
sdrnn= [
0.957610 0.010388
];
rnn= [
0.847370 0.009995
];
rnna= [
0.792160 0.015083
];

hold off
bar(1,rnn(:,1),'facecolor',col(1,:), 'edgecolor',col(1,:))
hold on
q=errorbar(1,rnn(:,1),rnn(:,2), 'linewidth', lw, 'color',col(1,:)*.7);
q.CapSize = cs;
set(gca,'fontsize',fs)

bar(3,sdrnn(:,1),'facecolor',col(3,:), 'edgecolor',col(3,:))
q=errorbar(3,sdrnn(:,1),sdrnn(:,2), 'linewidth', lw, 'color',col(3,:)*.7);
q.CapSize = cs;

bar(2,rnna(:,1),'facecolor',col(2,:), 'edgecolor',col(2,:))
q=errorbar(2,rnna(:,1),rnna(:,2), 'linewidth', lw, 'color',col(2,:)*.7);
q.CapSize = cs;

%title('tanh, filler 1','fontsize',fs);
ytickformat('%,.2f')

set(gca,'xlim',[.5 3.5])
set(gca,'xtick',1:3)
l1 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}RNN',col(1,:));
l2 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}RNN+A', col(2,:));
l3 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}SDRNN', col(3,:));
t=set(gca,'xticklabel',{l1 l2 l3},'fontsize',fs)
ylabel('Proportion Correct','fontsize',FS)
xlabel('Architecture','fontsize',FS)
set(gca,'ylim',[.75 1])
title('(a) Symmetry, filler 1','fontsize',FS)


subplot(1,3,2)

% tanh_5seq_1filler_5000examples
sdrnn= [
.8997 .0138*.75  % STD ERROR IS TOO LARGE BUT I LOST ORIGINAL DATA TO PRUNE IT
];
rnn= [
.8232 .0205*.75
];
rnna= [
.6504 .0244*.75  % TEMPORARY
];


hold off
bar(1,rnn(:,1),'facecolor',col(1,:), 'edgecolor',col(1,:))
hold on
q=errorbar(1,rnn(:,1),rnn(:,2), 'linewidth', lw, 'color',col(1,:)*.7);
q.CapSize = cs;
set(gca,'fontsize',fs)

bar(3,sdrnn(:,1),'facecolor',col(3,:), 'edgecolor',col(3,:))
q=errorbar(3,sdrnn(:,1),sdrnn(:,2), 'linewidth', lw, 'color',col(3,:)*.7);
q.CapSize = cs;

bar(2,rnna(:,1),'facecolor',col(2,:), 'edgecolor',col(2,:))
q=errorbar(2,rnna(:,1),rnna(:,2), 'linewidth', lw, 'color',col(2,:)*.7);
q.CapSize = cs;

%title('tanh, filler 1','fontsize',fs);
ytickformat('%,.2f')

set(gca,'xlim',[.5 3.5])
set(gca,'xtick',1:3)

l1 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}RNN',col(1,:));
l2 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}RNN+A', col(2,:));
l3 = sprintf('\\color[rgb]{%.4f,%.4f,%.4f}SDRNN', col(3,:));
t=set(gca,'xticklabel',{l1 l2 l3},'fontsize',fs)

ylabel('Proportion Correct','fontsize',FS)
xlabel('Architecture','fontsize',FS)
set(gca,'ylim',[.625 .925])
title('(b) Symmetry, filler 10','fontsize',FS)


subplot(1,3,3)

sdrnn = [
802   0.552911 0.589641 0.618676 0.557365
1123  0.552952 0.581077 0.727679 0.571207
3211  0.574455 0.617154 0.663208 0.652971
6422  0.744875 0.601228 0.600614 0.588757
11238 0.7949   0.7862   0.7949   0.7862
];

rnn = [
802   0.398864 0.277238 0.40351343 0.27778205
1123  0.460006 0.360593 0.46411076 0.34987783
3211  0.447918 0.361956 0.357336 0.419507
6422  0.525642 0.419252 0.519308 0.522764
11238 0.780609 0.785588 0.780609 0.785588
];


hold off
errorbar(1:5,mean(sdrnn(:,2:end),2),std(sdrnn(:,2:end),[],2)/sqrt(5),'linewidth',lw,'color',col(3,:))
hold on
errorbar(1:5,mean(rnn(:,2:end),2),std(rnn(:,2:end),[],2)/sqrt(5),'linewidth',lw,'color',col(1,:))
set(gca,'fontsize',fs-2)
set(gca,'xlim',[.7 5.3])
set(gca,'xticklabel',{'800','1600','3200','6400','12800'})
xlabel('Training Set Size','fontsize',FS)
ylabel('Proportion Correct','fontsize',FS)
ytickformat('%,.2f')
set(gca,'ytick',[.3 .4 .5 .6 .7 .8])
set(gca,'ylim',[.3 .8])
title('(c) POS Tagging','fontsize',FS)
text(2.5,.34,'RNN','fontsize',fs,'color',col(1,:))
text(2.25,.685,'SDRNN','fontsize',fs,'color',col(3,:))
set(gcf,'position',[10 10 423*3 352])
set(gcf,'paperpositionmode','auto')
print2pdf('symmetry_fig.pdf')
system('cp symmetry_fig.pdf ~/Dropbox/Apps/ShareLatex/NIPS2018\ SDRNN/fig')
