
function PrintNice(f)
    set(gcf,'Units','inches');
    screenposition = get(gcf,'Position');
    papersize = screenposition(3:4);
    set(gcf,...
        'PaperPosition',[0 0 papersize],...
        'PaperSize',papersize);
    

end
