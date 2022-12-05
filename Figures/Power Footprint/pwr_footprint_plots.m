%% Normal versionn
legend_entries = cell(9,1);

close all
f = figure(1);
hold on;


% Change color of the last two plots (matlab only orders until 7!
%      RGB Triplet	   Appearance
colors = {[0.0000 0.4470 0.7410], ... % dark blue
          [0.8500 0.3250 0.0980], ... % dark orange
          [0.9290 0.6940 0.1250], ... % dark yellow
          [0.4940 0.1840 0.5560], ... % dark purple
          [0.4660 0.6740 0.1880], ... % medium green
          [0.3010 0.7450 0.9330], ... % light blue
          [0.6350 0.0780 0.1840], ... % dark red
          [0.6275 0.3216 0.1765], ... % INVENTED 2 - sienabrown
          [0.7000 0.3640 0.7060], ... % INVENTED 1 - light purple
          %[0.8235 0.4118 0.1176], ... % chocbrown 
          %[0.5451 0.2706 0.0745], ... % saddlebrown
          };
      
for i = 1:9
    data = load(['scen_' num2str(i) '.mat']).data;
    plot(data, '-o', 'Color', colors{i}, ...
         'MarkerSize', 3, 'MarkerFaceColor', 'auto',...
         'Linewidth', 1.2);
    legend_entries{i} = ['Scenario ' num2str(i)];
end

axis tight;
% legend(legend_entries, 'Location', 'northwest', 'NumColumns', 1)
legend(legend_entries, 'Position', [0.19, 0.645, 0.1, 0.2], 'NumColumns', 1)
xlabel('Beam index')
ylabel('Normalized power')

range = 33-30:10:33+30;
xticks(range);
labels = strsplit(num2str(range - 33));
labels{floor(length(range)/2)+1} = 'best';
xticklabels(labels);
grid on
box on
ax = gca(); ax.GridAlpha = 0.3;
saveas(f, 'power_footprint.svg')
print(f, 'power_footprint', '-depsc')

%% Zoomed in versionn


legend_entries = cell(9,1);

close all
f = figure(1);
hold on;
for i = 1:9
    data = load(['scen_' num2str(i) '.mat']).data;
    p = plot(data, '-o', 'MarkerSize', 5, 'MarkerFaceColor', 'auto');
    %filled markers 
    % set(p, 'MarkerFaceColor', get(p,'Color')); % 'auto' for white
    legend_entries{i} = ['scen ' num2str(i)];
end

axis tight;
legend(legend_entries, 'Location', 'northwest', 'NumColumns', 1)
xlabel('Beam index')
ylabel('Normalized power')

m = 5;
range = 33-m:33+m;
xlim([min(range), max(range)]);
ylim([0.7,1]);
xticks(range);
labels = strsplit(num2str(range - 33));
labels{m+1} = 'best';
xticklabels(labels);
grid on;
ax = gca(); ax.GridAlpha = 0.3;
print(f, 'zoomed', '-depsc')
saveas(f, 'zoomed.eps')