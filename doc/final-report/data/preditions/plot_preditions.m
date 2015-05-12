% #!/usr/bin/env octave -qf

% figure('visible','off');

fig=figure;

model_names = ['Ensamble'; '  KNN   '; '  NuSVR '; '   GB   ';];

gt = load('GT.txt');

for i=1:size(model_names)
    filename = strcat(strtrim(model_names(i,:)), '.txt');
    
    p = load(filename);    

    hold off;
    plot(p, 'g', 'LineWidth', 8, 'Color', [0,0.5,0]);
    hold on;
    plot(gt, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

    xlabel('Instances', 'FontSize', 24, 'FontName', 'Helvetica');
    ylabel('Revenue', 'FontSize', 24, 'FontName', 'Helvetica');

    set(gca, 'FontSize', 24, 'FontName', 'Helvetica');

    h = legend({'Prediction', 'Ground Truth'}, 'Location', 'northeast');

    ylim([0, 2.5e7]);

    set(h, 'FontSize', 24, 'FontName', 'Helvetica');

    output = strcat('../../figs/', strtrim(model_names(i,:)), '_predit.eps');
%     fprintf(stdout, 'saving to %s\n', output);
    set(gcf,'paperposition',[0,0,8*1.5,5.0*1.5])
    print('-depsc', output);
end


% bar(scores);

% % title('RMSE / 10^6 ', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Helvetica');


% ylim([1.6 2.5]);

% set(gca, 'XTickLabel', model_names);

% h = legend({'Cross Validation', 'Public Board'});

% set(h, 'FontSize', 24, 'FontName', 'Helvetica');

% set(gca, 'FontSize', 24, 'FontName', 'Helvetica');

% print('-depsc', '../figs/models.eps', '-S800,400');