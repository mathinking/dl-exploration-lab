function plotClassProportion(labelVerified, verifiedCubes, totalVolume)
    propVerifiedVolumePerClass = zeros(1,5);
    for idxLabel = 1:5
        predictClassIdx = (labelVerified == idxLabel);
        verifiedCubeVolumePerClass = sum(prod( ...
            verifiedCubes.xUpper(2:3, predictClassIdx) - ...
            verifiedCubes.xLower(2:3, predictClassIdx)));
        propVerifiedVolumePerClass(idxLabel) = verifiedCubeVolumePerClass / totalVolume;
    end
    propVerifiedVolumePerClass(end+1) = 1 - sum(propVerifiedVolumePerClass);
    figure
    b = bar([helper.classNames; categorical("Unverified")], ...
        extractdata(propVerifiedVolumePerClass));
    b.FaceColor = "flat";
    for idx = 1:5
        b.CData(idx,:) = helper.getClassColors(idx);
    end
    b.CData(6,:) = [0.5 0.5 0.5];
    ylim([0 1])
    text(1:length(propVerifiedVolumePerClass), ...
        extractdata(propVerifiedVolumePerClass), ...
        num2str(extractdata(propVerifiedVolumePerClass)',"%.2f"), ...
        vert="bottom",horiz="center");
    title("Proportion of ODD by Class")
end
