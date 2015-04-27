#!/usr/bin/env octave -qf

figure('visible','off');

s = load('scores.txt');
s = s / 1e6;
scatter(s(:,1), s(:,2), 30, 'filled');

title('RMSE / 10^6 ', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Cross-Validation', 'FontSize', 16, 'FontName', 'Helvetica');
ylabel('Public Board', 'FontSize', 16, 'FontName', 'Helvetica');

ylim([1.6 2.0])

set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

print -depsc scores.eps