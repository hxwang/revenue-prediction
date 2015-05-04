#!/usr/bin/env octave -qf

figure('visible','off');

s = load('best_scores.txt');
s = sort(s) / 1e6;
% scatter(s(:,1), s(:,2), 30, 'filled');
plot(s, 'LineWidth', 8);

grid on;

title('Public Board', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Rank', 'FontSize', 16, 'FontName', 'Helvetica');
ylabel('RMSE / 10^6', 'FontSize', 16, 'FontName', 'Helvetica');

ylim([1.4 3.0])

set(gca, 'FontSize', 16, 'FontName', 'Helvetica');

print -depsc scores.eps