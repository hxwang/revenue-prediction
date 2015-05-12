% #!/usr/bin/env octave -qf

% figure('visible','off');

revenues = load('revenue.txt');

hist(revenues, 50);

% title('RMSE / 10^6 ', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Revenue', 'FontSize', 16, 'FontName', 'Helvetica');
ylabel('Count', 'FontSize', 16, 'FontName', 'Helvetica');

% ylim([1.6 2.0])

set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

set(gcf,'paperposition',[0,0,8*1.5,3*1.5])

print('-depsc', '../../doc/final-report/figs/revenue.eps');