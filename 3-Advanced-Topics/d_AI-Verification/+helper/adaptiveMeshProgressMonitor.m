function adaptiveMeshProgressMonitor
    figure
    axis equal
    xlim([-pi pi])
    ylim([-pi pi])
    xticks(-pi:pi/4:pi)
    yticks(-pi:pi/4:pi)
    tickLabels = {'$-\pi$','$-\frac{3\pi}{4}$','$-\frac{\pi}{2}$','$-\frac{\pi}{4}$', ...
        '$0$','$\frac{\pi}{4}$','$\frac{\pi}{2}$','$\frac{3\pi}{4}$','$\pi$'};
    xticklabels(tickLabels)
    yticklabels(tickLabels)
    set(gca,TickLabelInterpreter="latex");
    xlabel("$\theta$",Interpreter="latex")
    ylabel("$\psi$",Interpreter="latex")
    title("Stable Regions of ODD")
    grid on
    for ii = 1:5
        hold on
        plot(-100,-100,"square", ...
            Color=helper.getClassColors(ii), ...
            MarkerFaceColor=helper.getClassColors(ii), ...
            MarkerEdgeColor=helper.getClassColors(ii));
    end
    hold off
    legend(helper.classNames,Location="eastoutside");
end
