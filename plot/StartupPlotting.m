% Default plot formatting - text interpreter/colors
%
% Function changes the text interpreter to the specified string, with the
% default as 'latex', and sets default plotting colors.
%
% Author:   Nicholas Barbara
% Email:    nicholas.barbara@sydney.edu.au

function StartupPlotting(str)

    % Inputs
    if nargin < 1
        str = 'latex';
    end

    % Change the defaults
    set(groot,'defaulttextinterpreter',str);
    set(groot,'defaultAxesTickLabelInterpreter',str);
    set(groot,'defaultLegendInterpreter',str);

    % Set colour scheme
    colors = NiceColors(8);
    set(groot,'defaultAxesColorOrder',[colors{1};colors{2};...
        colors{3};colors{4};colors{5};colors{6};colors{7};colors{8}]);
end

% Helper function to get nice colors
function colorCell = NiceColors(numColors)

    if numColors > 8
        error("Too many colors for this colorscheme.");
    end
    
    colorCell = GiveMeColors(numColors,'wong');
end
