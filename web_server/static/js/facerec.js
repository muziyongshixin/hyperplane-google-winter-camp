$(function() {
    function getQueryStrings() {
        var vars = [], hash, hashes;
        if (window.location.href.indexOf('#') === -1) {
            hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
        } else {
            hashes = window.location.href.slice(window.location.href.indexOf('?') + 1, window.location.href.indexOf('#')).split('&');
        }
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
                    var score = res.scores[i][j];
                    var color = getColorForPercentage(score / 100.);
                    var placement = j < numB / 2 ? 'right' : 'left';
                    html += '<td style="background: ' + color + '" ' +
                                'data-content="<img src=' + res.faces_a[i] + '> - <img src=' + res.faces_b[j] + '>" '+
                                'data-placement="' + placement + '">' +
                              score.toFixed(2) +
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
        $('#feature_model').val(res.feature_model);
        $('#result .det-model').html('detect model: ' + res.detect_model);
        $('#detect_model').val(res.detect_model);
        if (res.detect_options) {
            $('#result .det-options').html('detect options: ' + JSON.stringify(res.detect_options));
            $('#min_face').val(res.detect_options.min_face);
        }
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

    $('#detect_model').on('change', function() {
        if ($('#detect_model').val().indexOf('mobile') != -1) {
            $('#min_face').val(70);
        } else {
            $('#min_face').val(0);
        }
    });

    $('form').on('submit', function() {
        $(this).find('.btn-submit').attr('disabled', 'disabled').html('Submitting...');
        return true;
    });
});
