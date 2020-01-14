$(function() {
    function getQueryStrings() {
        var vars = [], hash;
        var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
        for(var i = 0; i < hashes.length; i++) {
            hash = hashes[i].split('=');
            vars.push(hash[0]);
            vars[hash[0]] = hash[1];
        }
        return vars;
    }

    var percentColors = [
        { pct: 0.0, color: { r: 0x00, g: 0xff, b: 0 } },
        { pct: 0.5, color: { r: 0xff, g: 0xff, b: 0 } },
        { pct: 1.0, color: { r: 0xff, g: 0x00, b: 0 } }
    ];

    var getColorForPercentage = function(pct) {
        for (var i = 1; i < percentColors.length - 1; i++) {
            if (pct < percentColors[i].pct) {
                break;
            }
        }
        var lower = percentColors[i - 1];
        var upper = percentColors[i];
        var range = upper.pct - lower.pct;
        var rangePct = (pct - lower.pct) / range;
        var pctLower = 1 - rangePct;
        var pctUpper = rangePct;
        var color = {
            r: Math.floor(lower.color.r * pctLower + upper.color.r * pctUpper),
            g: Math.floor(lower.color.g * pctLower + upper.color.g * pctUpper),
            b: Math.floor(lower.color.b * pctLower + upper.color.b * pctUpper)
        };
        return 'rgba(' + [color.r, color.g, color.b].join(',') + ',0.5)';
    }

    function renderResult(res) {
        var html = '',
            numA = res.faces_a.length,
            numB = res.faces_b.length;
        var alpha = alpha_beta[res.feature_model][0],
            beta = alpha_beta[res.feature_model][1];
        for (var i = -1; i < numA; ++i) {
            if (i == -1) {
                html += '<thead>'
            } else if (i == 0) {
                html += '<tbody>'
            }
            html += '<tr>'
            for (var j = -1; j < numB; ++j) {
                if (i == -1 && j == -1) {
                    html += '<th>A \\ B</th>';
                } else if (i == -1) {
                    html += '<th><img src="' + res.faces_b[j] + '"><br>' + (j + 1) + '</th>';
                } else if (j == -1) {
                    html += '<th><img src="' + res.faces_a[i] + '">' + (i + 1) + '</th>';
                } else {
                    var normed = 100.0 / (1. + Math.exp(alpha * res.pair_dists[i][j] + beta));
                    var color = getColorForPercentage(normed / 100.);
                    var placement = j < numB / 2 ? 'right' : 'left';
                    html += '<td style="background: ' + color + '" ' +
                                'data-content="<img src=' + res.faces_a[i] + '> - <img src=' + res.faces_b[j] + '>" '+
                                'data-placement="' + placement + '">' +
                              res.pair_dists[i][j].toFixed(2) + ' / ' + normed.toFixed(2) +
                            '</td>';
                }
            }
            html += '</tr>';
            if (i == -1) {
                html += '</thead>'
            } else if (i == numA - 1) {
                html += '</tbody>'
            }
        }
        $('#result .feat-model').html('feature model: ' + res.feature_model);
        $('#result .det-model').html('detect model: ' + res.detect_model);
        $('#result .alpha').html(alpha);
        $('#result .beta').html(beta);
        $('#result table').html(html);
        $('#result').show();

        $('#result td').popover({
            'html': true,
            'trigger': 'hover',
            'animation': false
        })
    }

    var queryStrings = getQueryStrings();
    var rid = queryStrings['r'];
    if (rid) {
        var resultUrl = 'results/' + rid + '/result.json'
        $.getJSON(resultUrl, function(res) {
            renderResult(res);
        });
    }

    $('form').on('submit', function() {
        $(this).find('.btn-submit').attr('disabled', 'disabled').html('Submitting...');
        return true;
    });
});
