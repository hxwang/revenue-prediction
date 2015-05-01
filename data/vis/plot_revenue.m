#!/usr/bin/env octave -qf

figure('visible','off');

s = dlmread('../train_scaled.csv');

hist(s(:,42), 50);

% title('RMSE / 10^6 ', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica');
% xlabel('Cross-Validation', 'FontSize', 16, 'FontName', 'Helvetica');
% ylabel('Public Board', 'FontSize', 16, 'FontName', 'Helvetica');

% ylim([1.6 2.0])

set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

print('-depsc', '../../report/figs/revenue.eps', '-S800,300');