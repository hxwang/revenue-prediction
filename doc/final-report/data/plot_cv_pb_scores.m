% #!/usr/bin/env octave -qf
% figure('visible','off');

f = figure;

s = load('scores.txt');
s = s / 1e6;
plot(s(:,1), s(:,2), 'bo', 'MarkerSize', 12, 'MarkerFaceColor', 'b');

grid on;

% title('RMSE / 10^6 ', 'FontSize', 20, 'FontWeight', 'bold', 'FontName', 'Helvetica');
xlabel('Cross-Validation', 'FontSize', 20, 'FontName', 'Helvetica');
ylabel('Public Board', 'FontSize', 20, 'FontName', 'Helvetica');

h = legend('RMSE');

set(h, 'FontSize', 20, 'FontName', 'Helvetica');

ylim([1.6 2.0])

set(gca, 'FontSize', 20, 'FontName', 'Helvetica');

set(gcf,'paperposition',[0,0,8*1.5,4.0*1.5])

print('-depsc', '../figs/cv_pb_scores.eps');

corrcoef(s(:,1), s(:,2))

close(f);