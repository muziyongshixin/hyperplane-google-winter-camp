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

    function toAngle(x) {
        return (x / 3.14159265 * 180.).toFixed(2);
    }

    function renderResult(res) {
        // console.log('click')
        var html = '';
        var records = res.records;
        var table = document.getElementById("result-table");

        for (var i = 0; i < records.length; i++) {
            record = records[i];
            var head = document.createElement("tr");
            table.appendChild(head);

            var th1 = document.createElement("th");
            var th1_text = document.createTextNode("Query");
            th1.appendChild(th1_text);
            head.appendChild(th1);

            var th2 = document.createElement("th");
            var th2_text = document.createTextNode("Reference");
            th2.appendChild(th2_text);
            head.appendChild(th2);

            var th3 = document.createElement("th");
            var th3_text = document.createTextNode("Start");
            th3.appendChild(th3_text);
            head.appendChild(th3);

            var th4 = document.createElement("th");
            var th4_text = document.createTextNode("End");
            th4.appendChild(th4_text);
            head.appendChild(th4);

            var query_tr = document.createElement("tr");
            table.appendChild(query_tr);

            // show result
            var result_num = record.video_nums;
            for (var x = 0; x < result_num; x++){
                var tr = document.createElement("tr");
                table.appendChild(tr);

                if (x == 0) {
                    var query_td = document.createElement("td");
                    var query_link = document.createElement("video");

                    query_link.height = 250;
                    query_link.width = 300;
                    query_link.src = record.query_video;
                    query_link.controls = true;
                    query_link.preload = "metadata";
                    // query_link.setAttribute('data-setup', '{}');
                    // query_link.setAttribute('height', 250);
                    // query_link.setAttribute('width', 300);
                    //
                    // query_link.setAttribute('className', "video-js vjs-default-skin");
                    // query_link.setAttribute("data-setup", "{}");
                    // var sourceMP4 = document.createElement("source");
                    // sourceMP4.src = record.query_video;
                    // sourceMP4.type = "video/mp4";
                    // query_link.appendChild(sourceMP4);
                    // var sourceFLV = document.createElement("source");
                    // sourceFLV.type = "video/x-flv";
                    // sourceFLV.src = record.query_video;
                    // query_link.appendChild(sourceFLV);
                    // var sourceobj = document.createElement("object");
                    // sourceobj.data = record.query_video;
                    // var emb = document.createElement("embed");
                    // emb.src = record.query_video;
                    // sourceobj.appendChild(emb);
                    // query_link.appendChild(sourceobj);

                    query_td.appendChild(query_link);
                    tr.appendChild(query_td);
                }
                else{
                    var query_td = document.createElement("td");
                    tr.appendChild(query_td);
                }

                var cur_video = record.result.video_lists[x];

                var td1 = document.createElement("td");
                    // td元素添加在tr元素后
                tr.appendChild(td1);
                var v_link = document.createElement("img");
                v_link.height = 300;
                v_link.width = 300;
                v_link.src = cur_video.url;
                v_link.controls = true;
                v_link.preload = "metadata";
                td1.appendChild(v_link);

                var td3 = document.createElement("td");
                    // td元素添加在tr元素后
                tr.appendChild(td3);
                var start = document.createElement('start');
                var start_text = document.createTextNode(cur_video.start);
                start.appendChild(start_text);
                td3.appendChild(start);

                var td4 = document.createElement("td");
                    // td元素添加在tr元素后
                tr.appendChild(td4);
                var end = document.createElement('end');
                var end_text = document.createTextNode(cur_video.end);
                end.appendChild(end_text);
                td4.appendChild(end);
            }
        }

        $('#result').show()
    }

    var queryStrings = getQueryStrings();
    var rid = queryStrings['r'];
    if (rid) {
        var resultUrl = 'results/search/' + rid + '/result.json';
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
