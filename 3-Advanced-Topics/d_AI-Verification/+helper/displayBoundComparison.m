function displayBoundComparison(ylCrown, yuCrown, ylAlpha, yuAlpha)
    classLabels = ["CoC", "WL", "WR", "SL", "SR"];
    ylC = extractdata(ylCrown);
    yuC = extractdata(yuCrown);
    ylA = extractdata(ylAlpha);
    yuA = extractdata(yuAlpha);

    figure
    hold on
    for ii = 1:5
        fill([ii-0.3 ii+0.3 ii+0.3 ii-0.3], ...
            [ylC(ii) ylC(ii) yuC(ii) yuC(ii)], ...
            [0.2 0.4 0.8], FaceAlpha=0.4, EdgeColor=[0.2 0.4 0.8], LineWidth=1.5);
        fill([ii-0.15 ii+0.15 ii+0.15 ii-0.15], ...
            [ylA(ii) ylA(ii) yuA(ii) yuA(ii)], ...
            [0.8 0.2 0.2], FaceAlpha=0.5, EdgeColor=[0.8 0.2 0.2], LineWidth=1.5);
    end
    hold off
    xticks(1:5)
    xticklabels(classLabels)
    ylabel("Network Output Score")
    title("Output Bound Comparison: CROWN vs. \alpha-CROWN")
    legend(["CROWN", "\alpha-CROWN"], Location="best")
    grid on
end
