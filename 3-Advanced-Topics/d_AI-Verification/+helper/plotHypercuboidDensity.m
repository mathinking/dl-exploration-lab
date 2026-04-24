function plotHypercuboidDensity(hypercuboidCenters)
    figure
    nbins = 50;
    histogram2(hypercuboidCenters(2,:), hypercuboidCenters(3,:), nbins)
    xlim([-pi pi])
    ylim([-pi pi])
    xticks(-pi:pi/4:pi)
    yticks(-pi:pi/4:pi)
    tickLabels = ["$-\pi$","$-\frac{3\pi}{4}$","$-\frac{\pi}{2}$","$-\frac{\pi}{4}$", ...
        "$0$","$\frac{\pi}{4}$","$\frac{\pi}{2}$","$\frac{3\pi}{4}$","$\pi$"];
    xticklabels(tickLabels)
    yticklabels(tickLabels)
    set(gca,TickLabelInterpreter="latex");
    xlabel("$\theta$",Interpreter="latex")
    ylabel("$\psi$",Interpreter="latex")
    zlabel("Count")
    title("Hypercuboid Density")
end
