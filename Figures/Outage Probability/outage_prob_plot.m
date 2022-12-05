%% Plot outage probability for 9 scenarios


x =    [80	    90	    95	    99];
data = [0.969	0.953	0.938	0.875;
        0.953	0.938	0.922	0.891;
        0.891	0.844	0.781	0.328;
        0.891	0.813	0.719	0.313;
        0.953	0.922	0.891	0.797;
        0.953	0.938	0.922	0.875;
        0.906	0.844	0.781	0.469;
        0.938	0.906	0.859	0.688;
        0.922	0.891	0.859	0.750];

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

legend_entries = cell(9,1);  
f = figure();
hold on;
for i = 1:9
    plot(x, data(i,:), '-o', 'Color', colors{i}, ...
         'MarkerSize', 3, 'MarkerFaceColor', 'auto',...
         'Linewidth', 1.2);
    legend_entries{i} = ['Scenario ' num2str(i)];
end

axis tight;
% legend(legend_entries, 'Location', 'northwest', 'NumColumns', 1)
legend(legend_entries, 'Location', 'SW', 'NumColumns', 1)
xlabel('Guaranteed reliability (1 - Outage probability)')
ylabel('Overhead savings')
ylim([min(ylim()), 1])
xticks(x);
xticklabels(x);
grid minor;
box on;
ax = gca(); ax.GridAlpha = 0.5;
saveas(f, 'outage2.svg')
print(f, 'outage2', '-depsc')

    