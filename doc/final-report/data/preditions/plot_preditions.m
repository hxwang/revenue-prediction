#!/usr/bin/env octave -qf

figure('visible','off');

model_names = ['GB'; 'KNN'; 'NuSVR'; 'Ensamble'];

gt = load('GT.txt');
gt = gt / 1e6;

for i=1:size(model_names)
    filename = strcat(model_names(i,:), '.txt');
    
    p = load(filename);
    p = p/1e6;

    hold off;
    plot(p, 'g', 'LineWidth', 8, 'Color', [0,0.5,0]);
    hold on;
    plot(gt, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

    xlabel('Instances', 'FontSize', 24, 'FontName', 'Helvetica');
    ylabel('Revenue 10^6', 'FontSize', 24, 'FontName', 'Helvetica');

    set(gca, 'FontSize', 24, 'FontName', 'Helvetica');

    h = legend({'Prediction', 'Groud Truth'}, 'Location', 'northeast');

    ylim([0, 25]);

    set(h, 'FontSize', 24, 'FontName', 'Helvetica');

    output = strcat('../../figs/', model_names(i,:), '_predit.eps');
    fprintf(stdout, 'saving to %s\n', output);
    print('-depsc', output, '-S800,540');
end


% bar(scores);

% % title('RMSE / 10^6 ', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Helvetica');


% ylim([1.6 2.5]);

% set(gca, 'XTickLabel', model_names);

% h = legend({'Cross Validation', 'Public Board'});

% set(h, 'FontSize', 24, 'FontName', 'Helvetica');

% set(gca, 'FontSize', 24, 'FontName', 'Helvetica');

% print('-depsc', '../figs/models.eps', '-S800,400');