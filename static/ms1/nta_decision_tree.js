var jobid = JSON.parse(document.getElementById('jobid').textContent);

// set path for webApp reduced CSV
var csv_path = '../Reduced_WebApp_input_file_perc.csv';

// set default threshold values
var thresholdData = {
  "repA": 66.7,
  "cvA": 1.25,
  "repB": 50.0,
  "cvB": 0.80,
  "repMin": 0.0,
  "repMax": 100.0,
  "cvMin": 0.0,
  "cvMax": 5.0
};

// set up objects for keeping count of occurrence and feat level filters
var occ_counts_A = {
  'nTotal': 0,
  'nPresent': 0,
  'nMissing': 0,
  'nOverRep': 0,
  'nUnderRep': 0,
  'nOverCV': 0,
  'nUnderCV': 0,
  'nOverCVOverMRL': 0,
  'nOverCVUnderMRL': 0,
  'nUnderCVOverMRL': 0,
  'nUnderCVUnderMRL': 0
};
var occ_counts_B = {
  'nTotal': 0,
  'nPresent': 0,
  'nMissing': 0,
  'nOverRep': 0,
  'nUnderRep': 0,
  'nOverCV': 0,
  'nUnderCV': 0,
  'nOverCVOverMRL': 0,
  'nOverCVUnderMRL': 0,
  'nUnderCVOverMRL': 0,
  'nUnderCVUnderMRL': 0
};
var feat_counts_A = {
  'nTotal': 0,
  'nPresent': 0,
  'nMissing': 0,
  'nOverRep': 0,
  'nUnderRep': 0,
  'nOverCV': 0,
  'nUnderCV': 0,
  'nOverCVOverMRL': 0,
  'nOverCVUnderMRL': 0,
  'nUnderCVOverMRL': 0,
  'nUnderCVUnderMRL': 0
};
var feat_counts_B = {
  'nTotal': 0,
  'nPresent': 0,
  'nMissing': 0,
  'nOverRep': 0,
  'nUnderRep': 0,
  'nOverCV': 0,
  'nUnderCV': 0,
  'nOverCVOverMRL': 0,
  'nOverCVUnderMRL': 0,
  'nUnderCVOverMRL': 0,
  'nUnderCVUnderMRL': 0
};

// we must wrap the rest of the script in the d3.csv function because
// it is asyncronous, meaning that we can not return data from it
d3.csv(csv_path).then(function(data) {

  //// set up sliders/input boxes for thresholds
  // Replicate Threshold A
  var sliderRepA = document.getElementById("ThreshSliderRange_repA"),
    inputBoxRepA = document.getElementById("ThreshSliderNumber_repA");
  sliderRepA.oninput = function() {
    if (sliderRepA.value > thresholdData["repMax"]) {
      sliderRepA.value = thresholdData["repMax"];
    } else if (sliderRepA.value < thresholdData["repMin"]) {
      sliderRepA.value = thresholdData["repMin"];
    }
    inputBoxRepA.value = sliderRepA.value;

    thresholdData['repA'] = sliderRepA.value

    occ_counts_A = get_occ_counts(occ_counts_A, thresholdData['repA'], thresholdData['cvA']);
    feat_counts_A = get_feat_counts(feat_counts_A, thresholdData['repA'], thresholdData['cvA']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }
  inputBoxRepA.oninput = function() {
    if (inputBoxRepA.value > thresholdData["repMax"]) {
      inputBoxRepA.value = thresholdData["repMax"];
    } else if (inputBoxRepA.value < thresholdData["repMin"]) {
      inputBoxRepA.value = thresholdData["repMin"];
    }
    sliderRepA.value = inputBoxRepA.value;

    thresholdData['repA'] = sliderRepA.value;

    occ_counts_A = get_occ_counts(occ_counts_A, thresholdData['repA'], thresholdData['cvA']);
    feat_counts_A = get_feat_counts(feat_counts_A, thresholdData['repA'], thresholdData['cvA']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }

  // CV Threshold A
  var sliderCVA = document.getElementById("ThreshSliderRange_cvA"),
    inputBoxCVA = document.getElementById("ThreshSliderNumber_cvA");
  sliderCVA.oninput = function() {
    if (sliderCVA.value > thresholdData["cvMax"]) {
      sliderCVA.value = thresholdData["cvMax"];
    } else if (sliderCVA.value < thresholdData["cvMin"]) {
      sliderCVA.value = thresholdData["cvMin"];
    }
    inputBoxCVA.value = Number(sliderCVA.value);

    thresholdData['cvA'] = sliderCVA.value

    occ_counts_A = get_occ_counts(occ_counts_A, thresholdData['repA'], thresholdData['cvA']);
    feat_counts_A = get_feat_counts(feat_counts_A, thresholdData['repA'], thresholdData['cvA']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }
  inputBoxCVA.oninput = function() {
    if (inputBoxCVA.value > thresholdData["cvMax"]) {
      inputBoxCVA.value = thresholdData["cvMax"];
    } else if (inputBoxCVA.value < thresholdData["cvMin"]) {
      inputBoxCVA.value = thresholdData["cvMin"];
    }
    sliderCVA.value = Number(inputBoxCVA.value);

    thresholdData['cvA'] = sliderCVA.value

    occ_counts_A = get_occ_counts(occ_counts_A, thresholdData['repA'], thresholdData['cvA']);
    feat_counts_A = get_feat_counts(feat_counts_A, thresholdData['repA'], thresholdData['cvA']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }

  // Replicate Threshold B
  var sliderRepB = document.getElementById("ThreshSliderRange_repB"),
    inputBoxRepB = document.getElementById("ThreshSliderNumber_repB");
  sliderRepB.oninput = function() {
    if (sliderRepB.value > thresholdData["repMax"]) {
      sliderRepB.value = thresholdData["repMax"];
    } else if (sliderRepB.value < thresholdData["repMin"]) {
      sliderRepB.value = thresholdData["repMin"];
    }
    inputBoxRepB.value = Number(sliderRepB.value);

    thresholdData['repB'] = Number(sliderRepB.value)

    occ_counts_B = get_occ_counts(occ_counts_B, thresholdData['repB'], thresholdData['cvB']);
    feat_counts_B = get_feat_counts(feat_counts_B, thresholdData['repB'], thresholdData['cvB']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }
  inputBoxRepB.oninput = function() {
    if (inputBoxRepB.value > thresholdData["repMax"]) {
      inputBoxRepB.value = thresholdData["repMax"];
    } else if (inputBoxRepB.value < thresholdData["repMin"]) {
      inputBoxRepB.value = thresholdData["repMin"];
    }
    sliderRepB.value = Number(inputBoxRepB.value);

    thresholdData['repB'] = sliderRepB.value

    occ_counts_B = get_occ_counts(occ_counts_B, thresholdData['repB'], thresholdData['cvB']);
    feat_counts_B = get_feat_counts(feat_counts_B, thresholdData['repB'], thresholdData['cvB']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }

  // CV Threshold B
  var sliderCVB = document.getElementById("ThreshSliderRange_cvB"),
    inputBoxCVB = document.getElementById("ThreshSliderNumber_cvB");
  sliderCVB.oninput = function() {
    if (sliderCVB.value > thresholdData["cvMax"]) {
      sliderCVB.value = thresholdData["cvMax"];
    } else if (sliderCVB.value < thresholdData["cvMin"]) {
      sliderCVB.value = thresholdData["cvMin"];
    }
    inputBoxCVB.value = Number(sliderCVB.value);

    thresholdData['cvB'] = Number(sliderCVB.value)

    occ_counts_B = get_occ_counts(occ_counts_B, thresholdData['repB'], thresholdData['cvB']);
    feat_counts_B = get_feat_counts(feat_counts_B, thresholdData['repB'], thresholdData['cvB']);
    tableCreate(occ_counts_A, occ_counts_B, feat_counts_A, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }
  inputBoxCVB.oninput = function() {
    if (inputBoxCVB.value > thresholdData["cvMax"]) {
      inputBoxCVB.value = thresholdData["cvMax"];
    } else if (inputBoxCVB.value < thresholdData["cvMin"]) {
      inputBoxCVB.value = thresholdData["cvMin"];
    }
    sliderCVB.value = Number(inputBoxCVB.value);

    thresholdData['cvB'] = sliderCVB.value

    occ_counts_B = get_occ_counts(occ_counts_B, thresholdData['repB'], thresholdData['cvB']);
    feat_counts_B = get_feat_counts(feat_counts_B, thresholdData['repB'], thresholdData['cvB']);
    tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, true)
    createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
    createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
    createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
    createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')
  }

  // get counts based on current 
  occ_counts_A = get_occ_counts(occ_counts_A, thresholdData['repA'], thresholdData['cvA']);
  occ_counts_B = get_occ_counts(occ_counts_B, thresholdData['repB'], thresholdData['cvB']);
  feat_counts_A = get_feat_counts(feat_counts_A, thresholdData['repA'], thresholdData['cvA']);
  feat_counts_B = get_feat_counts(feat_counts_B, thresholdData['repB'], thresholdData['cvB']);

  tableCreate(occ_counts_A, feat_counts_A, occ_counts_B, feat_counts_B, false)
  createOccTree(occ_counts_A, 'occTreeABox', 'occTreeASVG')
  createOccTree(occ_counts_B, 'occTreeBBox', 'occTreeBSVG')
  createFeatTree(feat_counts_A, 'featTreeABox', 'featTreeASVG')
  createFeatTree(feat_counts_B, 'featTreeBBox', 'featTreeBSVG')

  // setup download buttons
  var buttonOccA = document.getElementById("downloadOccA");
  buttonOccA.addEventListener("click", function() {
    downloadSVG('occTreeASVG', 'occTreeABox', 'logicTreeA-occurrenceLevel.svg')
  }, false)

  var buttonFeatA = document.getElementById("downloadFeatA");
  buttonFeatA.addEventListener("click", function() {
    downloadSVG('featTreeASVG', 'featTreeABox', 'logicTreeA-featureLevel.svg')
  }, false)

  var buttonOccB = document.getElementById("downloadOccB");
  buttonOccB.addEventListener("click", function() {
    downloadSVG('occTreeBSVG', 'occTreeBBox', 'logicTreeB-occurrenceLevel.svg')
  }, false)

  var buttonFeatB = document.getElementById("downloadFeatB");
  buttonFeatB.addEventListener("click", function() {
    downloadSVG('featTreeBSVG', 'featTreeBBox', 'logicTreeB-featureLevel.svg')
  }, false)






  //////////////////////////////////////////////////
  ////////// Here be where functions live //////////
  //////////////////////////////////////////////////
  
  /**
   * Function for getting counts at occurrence level
   * @param {Object} occ_counts The occ_counts_X object whose data should be updated.
   * @param {Number} repThresh  The replicate threshold for occ_counts_X, e.g. occ_counts_A.
   * @param {Number} cvThresh   The CV threshold for occ_counts_X.
   * @returns {Object} The object of occurrence counts at different filter levels.
   */
  function get_occ_counts(occ_counts, repThresh, cvThresh) {
    // first we need to reset the counts in our occ_counts object
    for (i in occ_counts) {
      occ_counts[i] = 0;
    }

    // get 'sample suffixes'. Since we don't have a priori knowledge of the sample names, we need
    // to find them by looking for the N_Abun column names, which always end with a unique sample name
    var column_headers = Object.keys(data[0]); // array of all column headers
    var sample_names = []; // e.g., ['_MB', '_53_T', '_54_T', ...]
    for (let header_i in column_headers) {
      if (column_headers[header_i].slice(0, 6) === 'N_Abun') {
        sample_names.push(column_headers[header_i].slice(6));
      }
    }

    // now iterate over data by row
    for (let iRow in data) {
      var row = data[iRow];

      if (row['Feature_ID']) {
        // iterate over sample names (occurrences)
        for (let i in sample_names) {
          var sample_name = sample_names[i];
          occ_counts['nTotal'] += 1; 
          // check to see if the sample exists... N_Abun_* is > 0
          var n_abun_header = `N_Abun${sample_name}`;
          if (Number(row[n_abun_header]) > 0) {
            occ_counts['nPresent'] += 1;

            // now we need to check the replicate threshold
            var sample_rep_header = "Replicate_Percent" + sample_name;
            if (Number(row[sample_rep_header]) >= repThresh) {
              // pass replicate
              occ_counts['nOverRep'] += 1;

              // now we check the CV threshold
              var sample_cv_header = "CV" + sample_name;
              if (Number(row[sample_cv_header]) < cvThresh) {
                // pass CV (pass replicate-->pass CV)
                occ_counts['nUnderCV'] += 1;

                // check if this occurrence passes MRL check
                var mrl_threshold_header = "Blank_MDL"; // change to "MRL" for alex // change to "Blank_MDL for tyler"
                var sample_mean_header = "Mean" + sample_name;
                if (Number(row[sample_mean_header]) >= Number(row[mrl_threshold_header])) {
                  // pass MRL (pass replicate-->pass CV-->pass MRL)
                  occ_counts['nUnderCVOverMRL'] += 1;
                } else {
                  // fail MRL (pass replicate-->pass CV-->fail MRL)
                  occ_counts['nUnderCVUnderMRL'] += 1;
                }
              } else {
                // fail CV (pass replicate-->fail CV)
                occ_counts['nOverCV'] += 1;

                // check if this occurrence passes MRL check
                var mrl_threshold_header = "Blank_MDL"; // change to "MRL" for alex // change to "Blank_MDL for tyler"
                var sample_mean_header = "Mean" + sample_name;
                if (Number(row[sample_mean_header]) >= Number(row[mrl_threshold_header])) {
                  // pass MRL (pass replicate-->fail CV-->pass MRL)
                  occ_counts['nOverCVOverMRL'] += 1;
                } else {
                  // fail MRL (pass replicate-->fail CV-->fail MRL)
                  occ_counts['nOverCVUnderMRL'] += 1;
                }
              }
            } else {
              // we failed replicate check
              occ_counts['nUnderRep'] += 1;
            }

          // N_abun = 0
          } else {
            occ_counts['nMissing'] += 1;
          }
        }
      } // END OF INNER LOOP
    } // END OF OUTER LOOP
    
    // Now do QA checks to make sure that the sum of appropriate children add up to their parent's count
    if (occ_counts['nOverCVOverMRL'] + occ_counts['nOverCVUnderMRL'] !== occ_counts['nOverCV']) {
      console.warn("nOverCVOverMRL + nOverCVUnderMRL !== nOverCV");
    }
    if (occ_counts['nUnderCVOverMRL'] + occ_counts['nUnderCVUnderMRL'] !== occ_counts['nUnderCV']) {
      console.warn("nUnderCVOverMRL + nUnderCVUnderMRL !== nUnderCV");
    }
    if (occ_counts['nUnderCV'] + occ_counts['nOverCV'] !== occ_counts['nOverRep']) {
      console.warn("nUnderCV + nOverCV !== nOverRep");
    }
    if (occ_counts['nOverRep'] + occ_counts['nUnderRep'] !== occ_counts['nPresent']) {
      console.warn("nOverRep + nUnderRep !== nPresent");
    }
    if (occ_counts['nPresent'] + occ_counts['nMissing'] !== occ_counts['nTotal']) {
      console.warn("nPresent + nMissing !== nTotal");
    }

    return occ_counts;
  }

  /**
   * Function for getting counts at feature level
   * @param {Object} feat_counts The occ_counts_X object whose data should be updated.
   * @param {Number} repThresh  The replicate threshold for feat_counts_X, e.g. feat_counts_A.
   * @param {Number} cvThresh   The CV threshold for feat_counts_X.
   * @returns {Object} The object of feature counts at different filter levels.
   */
  function get_feat_counts(feat_counts, repThresh, cvThresh) {
    // first we need to reset the counts in our feat_counts object
    for (i in feat_counts) {
      feat_counts[i] = 0;
    }

    // we need to store information about "pass-hierarchy" to keep track of the 'highest-level-occurrence' in a feature.
    // e.g., if all occurrences of a feature are missing except for one occurrence that passes all filtering steps,
    //       then the feature is said to have passed.
    var pass_hierarchy = {
      'missing': 0,
      'present': 1,
      'underRep': 2,
      'overCV': 3,
      'overCVUnderMRL': 4,
      'overCVOverMRL' : 5,
      'underCV': 6,
      'underCVUnderMRL': 7,
      'underCVOverMRL': 8
    }

    // get 'sample suffixes'. Since we don't have a priori knowledge of the sample names, we need
    // to find them by looking for the N_Abun column names, which always end with a unique sample name
    var column_headers = Object.keys(data[0]); // array of all column headers
    var sample_names = []; // e.g., ['_MB', '_53_T', '_54_T', ...]
    for (let header_i in column_headers) {
      if (column_headers[header_i].slice(0, 6) === 'N_Abun') {
        sample_names.push(column_headers[header_i].slice(6));
      }
    }

    // we want to first check if the feature has a given occurrence that passes
    // both the replicate and CV checks (if 1 or more occurrence passes, the feature passes).
    // If the feature fails one of these, then we do nothing further.
    // If the feature passes replicate and CV checks, then we will check if the
    // feature passes the MRL check -- if any occurrence passes the MRL check, then
    // the feature passes the MRL check. To prevent ourselves from having to 
    // reiterate back over the occurrences of a feature, we will create an mrlPass 
    // flag variable whose value will only be used in the case that the feature
    // passes the replicate and CV thresholds.

    // iterate over rows (features)
    for (let iRow in data) {
      var row = data[iRow];

      // ensure we are not looking at an empty row at the end of CSV.
      if (row['Feature_ID']) {
        var max_pass = 'missing';
        var mrlPass = false;

        // iterate over sample names (occurrences)
        for (let i in sample_names) {
          var sample_name = sample_names[i];
          // check to see if the sample exists... N_Abun_* is > 0
          var n_abun_header = `N_Abun${sample_name}`;
          if (Number(row[n_abun_header]) > 0) {

            // update max_pass if needed (I think this not needed here, but for clarity)
            if (pass_hierarchy['present'] > pass_hierarchy[max_pass]) {
              max_pass = 'present';
            }

            // check if this occurrence within the feature passes MRL check (and hence causes the feature to pass)
            var mrl_threshold_header = "Blank_MDL"; // change to "MRL" for alex // change to "Blank_MDL for tyler"
            var sample_mean_header = "Mean" + sample_name;
            if (Number(row[sample_mean_header]) >= Number(row[mrl_threshold_header])) {
              mrlPass = true;
            }

            // now we need to check the replicate threshold
            var sample_rep_header = "Replicate_Percent" + sample_name;
            if (Number(row[sample_rep_header]) >= repThresh) {
              // now we check the CV threshold
              var sample_cv_header = "CV" + sample_name;
              if (Number(row[sample_cv_header]) < cvThresh) {
                if (pass_hierarchy['underCV'] > pass_hierarchy[max_pass]) {
                  max_pass = 'underCV';
                }
              } else {
                // if we failed CV check, we should update max_pass
                if (pass_hierarchy['overCV'] > pass_hierarchy[max_pass]) {
                  max_pass = 'overCV';
                }
              }
            } else {
              // If we failed replicate check, we should update max_pass
              if (pass_hierarchy['underRep'] > pass_hierarchy[max_pass]) {
                max_pass = 'underRep';
              }
            }
          }
        }

        // Now check for MRL IF the feature passed replicate and CV checks
        if (max_pass === 'underCV') {
          if (mrlPass === true) {
            max_pass = 'underCVOverMRL';
          } else {
            max_pass = 'underCVUnderMRL';
          }
        } else {
          if (mrlPass === true) {
            max_pass = 'overCVOverMRL';
          } else {
            max_pass = 'overCVUnderMRL';
          }
        }
      } // END OF FEATURE
      
      // now we can determine our counts by using the max_pass string
      feat_counts['nTotal'] += 1;
      if (max_pass === 'missing') {
        feat_counts['nMissing'] += 1;
      } else if (max_pass == 'present') {
        feat_counts['nPresent'] += 1;
      } else if (max_pass === 'underRep') {
        feat_counts['nPresent'] += 1;
        feat_counts['nUnderRep'] += 1;
      } else if (max_pass === 'overCVUnderMRL') {
        feat_counts['nPresent'] += 1;
        feat_counts['nOverRep'] += 1;
        feat_counts['nOverCV'] += 1;
        feat_counts['nOverCVUnderMRL'] += 1;
      } else if (max_pass === 'overCVOverMRL') {
        feat_counts['nPresent'] += 1;
        feat_counts['nOverRep'] += 1;
        feat_counts['nOverCV'] += 1;
        feat_counts['nOverCVOverMRL'] += 1;
      } else if (max_pass === 'underCVUnderMRL') {
        feat_counts['nPresent'] += 1;
        feat_counts['nOverRep'] += 1;
        feat_counts['nUnderCV'] += 1;
        feat_counts['nUnderCVUnderMRL'] += 1;
      } else if (max_pass === 'underCVOverMRL') {
        feat_counts['nPresent'] += 1;
        feat_counts['nOverRep'] += 1;
        feat_counts['nUnderCV'] += 1;
        feat_counts['nUnderCVOverMRL'] += 1;
      }
    } // END OF FEATURES
    // Now do QA checks to make sure that the sum of appropriate children add up to their parent's count
    if (feat_counts['nOverCV'] + feat_counts['nUnderCV'] !== feat_counts['nOverRep']) {
      console.warn("nOverCV + nUnderCV !== nOverRep");
    }
    if (feat_counts['nOverRep'] + feat_counts['nUnderRep'] !== feat_counts['nPresent']) {
      console.warn("nOverRep + nUnderRep !== nPresent");
    }
    if (feat_counts['nUnderCVOverMRL'] + feat_counts['nUnderCVUnderMRL'] !== feat_counts['nUnderCV']) {
      console.warn("nUnderCVOverMRL + nUnderCVUnderMRL !== nUnderCV");
    }
    if (feat_counts['nPresent'] + feat_counts['nMissing'] !== feat_counts['nTotal']) {
      console.warn("nPresent + nMissing !== nTotal");
    }
    if (feat_counts['nOverCVOverMRL'] + feat_counts['nOverCVUnderMRL'] !== feat_counts['nOverCV']) {
      console.warn("nOverCVOverMRL + nOverCVUnderMRL !== nOverCV");
    }
    return feat_counts;
  }

  /**
   * Function for generating table
   * @param {Object} occ_counts_A 
   * @param {Object} occ_counts_B 
   * @param {Object} feat_counts_A 
   * @param {Object} feat_counts_B 
   * @param {Bool} del If we want to delete the table before making a new one
   */
  function tableCreate(occ_counts_A, occ_counts_B, feat_counts_A, feat_counts_B, del=true) {
    // remove table if it exists
    var element = document.getElementById('tTable');
    if (del) {
        element.parentNode.removeChild(element);
    }

    objs = [occ_counts_A, occ_counts_B, feat_counts_A, feat_counts_B]

    // setup column/row labels
    var row_names = [
      'Total',
      'Present',
      'Missing',
      "< Replicate",
      "\u2265 Replicate",
      "> CV",
      "> CV \u21d2 < MRL",
      "> CV \u21d2 \u2265 MRL",
      "\u2264 CV",
      "\u2264 CV \u21d2 < MRL",
      "\u2264 CV \u21d2 \u2265 MRL"
    ];
    var col_names = [
      'Filter Label',
      'Occ A',
      'Feat A',
      'Occ B',
      'Feat B'
    ]

    // get the number of columns/rows
    const n_rows = row_names.length,
      n_columns = 5; // Filter; OccA; OccB; FeatA; FeatB

      // create <table> and <tbody> elements
      const tbl = document.createElement("table");
      tbl.setAttribute('id', 'tTable');
      const tblBody = document.createElement("tbody");
      
      // creating cells --> | Count_name | Count_value |
      for (let i = 0; i <= n_rows; i++) {
        // creates a table row
        const row = document.createElement("tr");

        for (let j = 0; j < n_columns; j++) {
          // create a <td> element and a text node, make the text for 
          // the node of <td> contents
          // if header, use <th> instead of <td>
          if (i === 0) {
            var cell = document.createElement("th");
            var cellText = document.createTextNode(col_names[j]);
          } else {
            var cell = document.createElement("td");
          }

          // if not first row, start adding data per row
          if (i !== 0) {
            // set data per column, first the filter label
            if (j === 0) {
              var cellText = document.createTextNode(row_names[i-1]);
            } else { // now other data dependant on our row
              if (i === 1) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nTotal']));
              } else if (i === 2) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nPresent']));
              } else if (i === 3) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nMissing']));
              } else if (i === 5) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nOverRep']));
              } else if (i === 4) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nUnderRep']));
              } else if (i === 6) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nOverCV']));
              } else if (i === 9) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nUnderCV']));
              } else if (i === 7) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nOverCVUnderMRL']));
              } else if (i === 8) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nOverCVOverMRL']));
              } else if (i === 10) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nUnderCVUnderMRL']));
              } else if (i === 11) {
                var cellText = document.createTextNode(numberWithCommas(objs[j-1]['nUnderCVOverMRL']));
              }
            }
          } 

          cell.appendChild(cellText);
          row.appendChild(cell);
        }
        // add row to the end of table body
        tblBody.appendChild(row);
      }

      // put <tbody> in table
      tbl.appendChild(tblBody);
      // append <table> into <body>
      document.getElementById('myTable').appendChild(tbl);
      // styles and attributes
      tbl.setAttribute("border", "2");
      var windowHeight = window.screen.height;
      tbl.setAttribute("height", windowHeight*0.5);
  }

  /**
   * 
   * @param {Object} feat_counts The object that has count data for the SVG tree you want to build.
   * @param {String} divID      The ID for the outer div that holds the SVG you want to build. 
   * @param {String} svgID      The ID for the SVG you want to build.  
   */
  function createFeatTree(feat_counts, divID, svgID) {
    // remove chart if it exists
    var chart = document.getElementById(divID);
    if (chart) {
      chart.childNodes[2].innerHTML = ''
    }
    // get dimensions of window
    var windowWidth = window.screen.width,
      windowHeight = window.screen.height,
      svgWidth = Math.min(windowWidth*0.42, 1600),
      svgHeight = windowHeight*0.55,
      svgWidth = 1400,
      svgHeight = 780;

    // get y-positions of each row of the decision tree
    var contentPaddingTopFactor = 0.22,
      contentPaddingBottomFactor = 0.08,
      contentPaddingTop = contentPaddingTopFactor * svgHeight,
      contentPaddingBottom = contentPaddingBottomFactor * svgHeight,
      yTitle = contentPaddingTop * 0.38,                
      yRow01 = contentPaddingTop,                    
      yRow04 = svgHeight - contentPaddingBottom,      
      yRow02 = yRow01 + ((yRow04 - yRow01) / 3),      
      yRow03 = yRow02 + ((yRow04 - yRow01) / 3); 

    // get x-positions of each box
    var xTitle = svgWidth * 0.02,
      xTotalSampleOccurrence = 0.9 * svgWidth / 3,           
      xMissing = 2 * svgWidth / 3;                    
      xOverReplicateThreshold = 1.2 * svgWidth / 7,     
      xUnderReplicateThreshold = 5.5 * svgWidth / 9,
      xUnderCVThreshold = svgWidth / 7,
      xOverCVThreshold = 5.4 * svgWidth / 9, 
      xUnderCVOverMRL = svgWidth / 18,
      xUnderCVUnderMRL = 3.9 * svgWidth / 13,
      xOverCVOverMRL = 7.2 *svgWidth / 13,
      xOverCVUnderMRL = 10.4 * svgWidth / 13;

    // create SVG element
    var svg = document.getElementById(svgID);
    svg.setAttribute('width', svgWidth);
    svg.setAttribute('height', svgHeight);

    var fontSizeText = "1.7rem";

    // add the title
    var textValue = 'Features   ' + svgID.charAt(svgID.length-4)
    var tag = `occ${svgID.charAt(svgID.length-4)}`; // for setting box ID on SVG children elements -- how arrows are drawn.
    svg = addTitleBox(svg, xTitle, yTitle, textValue, '3.0rem', 'rgba(0,0,0,0)', 'transparent', "white", `treeTitle${tag}`);

    // total sample occurrence text and box
    textValue = `Present &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nPresent']) + "  ", "#FFF", xTotalSampleOccurrence, yRow01, textValue, fontSizeText, "#FFF", "transparent", "white", `nPresent${tag}`);

    // missing text and box
    textValue = `Missing &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nMissing']) + "  ", "#FFF", xMissing, yRow01, textValue, fontSizeText, "#FFF", "#595959", "white", `nMissing${tag}`);

    // over replicate text and box
    textValue = `&#8805 Replicate Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nOverRep']) + "  ", "#FFF", xOverReplicateThreshold, yRow02, textValue, fontSizeText, "#FFF", "transparent", "white", `nOverRep${tag}`);

    // under replicate text and box
    textValue = `< Replicate Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nUnderRep']) + "  ", "#FFF", xUnderReplicateThreshold, yRow02, textValue, fontSizeText, "#FFF", "#595959", "white", `nUnderRep${tag}`);

    // under CV text and box
    textValue = `&#8804 CV Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nUnderCV']) + "  ", "#FFF", xUnderCVThreshold, yRow03, textValue, fontSizeText, "#FFF", "transparent", "white", `nUnderCV${tag}`);

    // over CV text and box
    textValue = `> CV Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nOverCV']) + "  ", "#FFF", xOverCVThreshold, yRow03, textValue, fontSizeText, "#F999A4", "#9f1d37", "white", `nOverCV${tag}`);

    // under CV over MRL
    textValue = `&#8805 MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nUnderCVOverMRL']) + "  ", "#FFF", xUnderCVOverMRL, yRow04, textValue, fontSizeText, "#FFF", "transparent", "white", `nUnderCVOverMRL${tag}`);

    // under CV under MRL
    textValue = `< MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nUnderCVUnderMRL']) + "  ", "#FFF", xUnderCVUnderMRL, yRow04, textValue, fontSizeText, "#FFF", "#595959", "white", `nUnderCVUnderMRL${tag}`);

    // over CV over MRL
    textValue = `&#8805 MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nOverCVOverMRL']) + "  ", "#FFF", xOverCVOverMRL, yRow04, textValue, fontSizeText, "#F999A4", "#9f1d37", "white", `nOverCVOverMRL${tag}`);

    // over CV under MRL
    textValue = `< MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(feat_counts['nOverCVUnderMRL']) + "  ", "#FFF", xOverCVUnderMRL, yRow04, textValue, fontSizeText, "#FFF", "#595959", "white", `nOverCVUnderMRL${tag}`);

    // add bifurcating arrows
    svg = addBifurcatingArrow(svg, `nPresent${tag}`, `nOverRep${tag}`, `nUnderRep${tag}`, 27, "#FFF", "#FFF", "#FFF", "#FFF", "#FFF", "#595959", "Kept", "Removed");
    svg = addBifurcatingArrow(svg, `nOverRep${tag}`, `nUnderCV${tag}`, `nOverCV${tag}`, 27, "#FFF", "#FFF", "#FFF", "#FFF", "#F999A4", "#9f1d37", "", "CV Flag");
    svg = addBifurcatingArrow(svg, `nUnderCV${tag}`, `nUnderCVOverMRL${tag}`, `nUnderCVUnderMRL${tag}`, 27, "#FFF", "#FFF", "#FFF", "#FFF", "#FFF", "#595959", "", "MRL Flag");
    svg = addBifurcatingArrow(svg, `nOverCV${tag}`, `nOverCVOverMRL${tag}`, `nOverCVUnderMRL${tag}`, 27, "#f999a4", "#9f1d37", "#f999a4", "#9f1d37", "#FFF", "#595959", "", "MRL Flag");

    // add text for Threshold values on SVG
    var replicateText = addText(30, 245, `<tspan text-decoration="underline">Replicate Threshold = ${thresholdData[`rep${tag.charAt(tag.length-1)}`]}%</tspan>`, 28, "white");
    svg.appendChild(replicateText);
    var replicateText = addText(40, 430, `<tspan text-decoration="underline">CV Threshold = ${thresholdData[`cv${tag.charAt(tag.length-1)}`]}</tspan>`, 28, "white");
    svg.appendChild(replicateText);

    // append svg to our container div
    document.getElementById(divID).appendChild(svg);
  }
  /**
   * 
   * @param {Object} occ_counts The object that has count data for the SVG tree you want to build.
   * @param {String} divID      The ID for the outer div that holds the SVG you want to build. 
   * @param {String} svgID      The ID for the SVG you want to build.  
   */
  function createOccTree(occ_counts, divID, svgID) {
    // remove chart if it exists
    var chart = document.getElementById(divID);
    if (chart) {
      chart.childNodes[2].innerHTML = ''
    }
    // get dimensions of window
    var windowWidth = window.screen.width,
      windowHeight = window.screen.height,
      svgWidth = Math.min(windowWidth*0.42, 1600),
      svgHeight = windowHeight*0.55,
      svgWidth = 1400,
      svgHeight = 780;

    // get y-positions of each row of the decision tree
    var contentPaddingTopFactor = 0.22,
      contentPaddingBottomFactor = 0.08,
      contentPaddingTop = contentPaddingTopFactor * svgHeight,
      contentPaddingBottom = contentPaddingBottomFactor * svgHeight,
      yTitle = contentPaddingTop * 0.38,                
      yRow01 = contentPaddingTop,                    
      yRow04 = svgHeight - contentPaddingBottom,      
      yRow02 = yRow01 + ((yRow04 - yRow01) / 3),      
      yRow03 = yRow02 + ((yRow04 - yRow01) / 3); 

    // get x-positions of each box
    var xTitle = svgWidth * 0.02,
      xTotalSampleOccurrence = 0.9 * svgWidth / 3,           
      xMissing = 2 * svgWidth / 3;                    
      xOverReplicateThreshold = 1.2 * svgWidth / 7,     
      xUnderReplicateThreshold = 5.5 * svgWidth / 9,
      xUnderCVThreshold = svgWidth / 7,
      xOverCVThreshold = 5.4 * svgWidth / 9, 
      xUnderCVOverMRL = svgWidth / 18,
      xUnderCVUnderMRL = 3.9 * svgWidth / 13,
      xOverCVOverMRL = 7.2 *svgWidth / 13,
      xOverCVUnderMRL = 10.4 * svgWidth / 13;

    // create SVG element
    var svg = document.getElementById(svgID);
    svg.setAttribute('width', svgWidth);
    svg.setAttribute('height', svgHeight);

    var fontSizeText = "1.7rem";

    // add the title
    var textValue = 'Occurrences   ' + svgID.charAt(svgID.length-4)
    var tag = `occ${svgID.charAt(svgID.length-4)}`; // for setting box ID on SVG children elements -- how arrows are drawn.
    svg = addTitleBox(svg, xTitle, yTitle, textValue, '3.0rem', 'rgba(0,0,0,0)', 'transparent', "white", `treeTitle${tag}`);

    // total sample occurrence text and box
    textValue = `Present &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nPresent']) + "  ", "#FFF", xTotalSampleOccurrence, yRow01, textValue, fontSizeText, "#FFF", "transparent", "white", `nPresent${tag}`);

    // missing text and box
    textValue = `Missing &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nMissing']) + "  ", "#FFF", xMissing, yRow01, textValue, fontSizeText, "#FFF", "#595959", "white", `nMissing${tag}`);

    // over replicate text and box
    textValue = `&#8805 Replicate Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nOverRep']) + "  ", "#FFF", xOverReplicateThreshold, yRow02, textValue, fontSizeText, "#FFF", "transparent", "white", `nOverRep${tag}`);

    // under replicate text and box
    textValue = `< Replicate Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nUnderRep']) + "  ", "#FFF", xUnderReplicateThreshold, yRow02, textValue, fontSizeText, "#FFF", "#595959", "white", `nUnderRep${tag}`);

    // under CV text and box
    textValue = `&#8804 CV Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nUnderCV']) + "  ", "#FFF", xUnderCVThreshold, yRow03, textValue, fontSizeText, "#FFF", "transparent", "white", `nUnderCV${tag}`);

    // over CV text and box
    textValue = `> CV Threshold &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nOverCV']) + "  ", "#FFF", xOverCVThreshold, yRow03, textValue, fontSizeText, "#F999A4", "#9f1d37", "white", `nOverCV${tag}`);

    // under CV over MRL
    textValue = `&#8805 MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nUnderCVOverMRL']) + "  ", "#FFF", xUnderCVOverMRL, yRow04, textValue, fontSizeText, "#FFF", "transparent", "white", `nUnderCVOverMRL${tag}`);

    // under CV under MRL
    textValue = `< MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nUnderCVUnderMRL']) + "  ", "#FFF", xUnderCVUnderMRL, yRow04, textValue, fontSizeText, "#FFF", "#595959", "white", `nUnderCVUnderMRL${tag}`);

    // over CV over MRL
    textValue = `&#8805 MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nOverCVOverMRL']) + "  ", "#FFF", xOverCVOverMRL, yRow04, textValue, fontSizeText, "#F999A4", "#9f1d37", "white", `nOverCVOverMRL${tag}`);

    // over CV under MRL
    textValue = `< MRL &nbsp&nbsp&nbsp <tspan fill="transparent"></tspan>`;
    svg = addBox(svg, "    " + numberWithCommas(occ_counts['nOverCVUnderMRL']) + "  ", "#FFF", xOverCVUnderMRL, yRow04, textValue, fontSizeText, "#FFF", "#595959", "white", `nOverCVUnderMRL${tag}`);

    // add bifurcating arrows
    svg = addBifurcatingArrow(svg, `nPresent${tag}`, `nOverRep${tag}`, `nUnderRep${tag}`, 27, "#FFF", "#FFF", "#FFF", "#FFF", "#FFF", "#595959", "Kept", "Removed");
    svg = addBifurcatingArrow(svg, `nOverRep${tag}`, `nUnderCV${tag}`, `nOverCV${tag}`, 27, "#FFF", "#FFF", "#FFF", "#FFF", "#F999A4", "#9f1d37", "", "CV Flag");
    svg = addBifurcatingArrow(svg, `nUnderCV${tag}`, `nUnderCVOverMRL${tag}`, `nUnderCVUnderMRL${tag}`, 27, "#FFF", "#FFF", "#FFF", "#FFF", "#FFF", "#595959", "", "MRL Flag");
    svg = addBifurcatingArrow(svg, `nOverCV${tag}`, `nOverCVOverMRL${tag}`, `nOverCVUnderMRL${tag}`, 27, "#f999a4", "#9f1d37", "#f999a4", "#9f1d37", "#FFF", "#595959", "", "MRL Flag");

    // add text for Threshold values on SVG
    var replicateText = addText(30, 245, `<tspan text-decoration="underline">Replicate Threshold = ${thresholdData[`rep${tag.charAt(tag.length-1)}`]}%</tspan>`, 28, "white");
    svg.appendChild(replicateText);
    var replicateText = addText(40, 430, `<tspan text-decoration="underline">CV Threshold = ${thresholdData[`cv${tag.charAt(tag.length-1)}`]}</tspan>`, 28, "white");
    svg.appendChild(replicateText);

    // append svg to our container div
    document.getElementById(divID).appendChild(svg);
  }

  /**
   * Generates an SVG text node for our tree diagrams.
   * @param {Number} x X position of text node.
   * @param {Number} y Y position of text node.
   * @param {String} value Text you want written in the node.
   * @param {String} fontSize 
   * @param {String} fontColor 
   * @returns SVG txt node.
   */
  function addText(x, y, value, fontSize, fontColor) {
    var txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttributeNS(null, "x", x);     
    txt.setAttributeNS(null, "y", y); 
    txt.setAttributeNS(null, "font-size", fontSize);
    txt.setAttribute('fill', fontColor);
    txt.innerHTML = value;
    txt.setAttribute("xml:space", "preserve");
    return txt;
  }

  /**
   * Generates an SVG rect element.
   * @param {Number} x X coordinate of rect element.
   * @param {Number} y Y coordinate of rect element.
   * @param {Number} width Width of rect element.
   * @param {Number} height Height of rect element.
   * @param {String} stroke Outline color of rect element.
   * @param {String} fill Fill color of rect element.
   * @param {Number} strokeWidth Border width for rect element.
   * @param {String} boxId The ID for the SVG box element.
   * @returns SVG rect element.
   */
  function addRectangle(x, y, width, height, stroke, fill, strokeWidth, boxId) {
    var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    rect.setAttribute('width', width);
    rect.setAttribute('height', height);
    rect.setAttribute('stroke', stroke);
    rect.setAttribute('fill', fill);
    rect.setAttribute('stroke-width', strokeWidth);
    rect.setAttribute('rx', "5");
    rect.setAttribute('id', boxId)
    return rect
  }

  /**
   * Adds a title to some SVG element.
   * @param {SVGElement} svgElement The SVG object you want 
   * @param {Number} xText X position of text.
   * @param {Number} yText Y position of text.
   * @param {String} valueText Text you want to be used as the title.
   * @param {String} fontSizeText 
   * @param {String} boxOutlineColor 
   * @param {String} boxFill 
   * @param {String} fontColor 
   * @param {String} boxId The ID for the title box element.
   * @returns Your SVG element with a title added.
   */
  function addTitleBox(svgElement, xText, yText, valueText, fontSizeText, boxOutlineColor, boxFill, fontColor, boxId) {
    var text = addText(xText, yText, valueText, fontSizeText, fontColor);
    svgElement.appendChild(text);
    var bbox = text.getBBox();
    var textWidth = bbox.width;
    var textHeight = bbox.height;
    var widthBox = textWidth * 1.1;
    var heightBox = textHeight * 1.8;
    var xBox = xText - (widthBox - textWidth)/2;
    var yBox = yText - textHeight - 0.11*heightBox;
    var rect = addRectangle(xBox, yBox, widthBox, heightBox, boxOutlineColor, boxFill, 2.5, boxId);
    svgElement.removeChild(text); // remove child then add again so it is on top... needed to append earlier to get dims
    svgElement.appendChild(rect);
    svgElement.appendChild(text);

    return svgElement;
  }

  /**
   * Adds a nested rect for highlighting counts in svg rect elements.
   * @param {SVGElement} svgElement SVG element you want to append.
   * @param {String} n The number (count) you want put in the box.
   * @param {String} nBGColor Background color of the box that the count is in.
   * @param {Number} xText 
   * @param {Number} yText 
   * @param {String} valueText 
   * @param {String} fontSizeText 
   * @param {String} boxOutlineColor 
   * @param {String} boxFill 
   * @param {String} fontColor 
   * @param {String} boxId Identifier for the box.
   * @returns Your SVG element with the appended box.
   */
  function addBox(svgElement, n, nBGColor, xText, yText, valueText, fontSizeText, boxOutlineColor, boxFill, fontColor, boxId) {
    // start by building the text box WITHOUT the count n value to get coordinates
    var textNoN = addText(xText, yText, valueText, fontSizeText, fontColor);
    svgElement.appendChild(textNoN);
    var bbox = textNoN.getBBox(),
      textNoNWidth = bbox.width,
      textNoNHeight = bbox.height,
      textNoNBoxHeight = 2 * textNoNHeight,
      xBoxRight = xText + (textNoNWidth),
      yBox = yText - textNoNBoxHeight - 0.11*textNoNBoxHeight;
    svgElement.removeChild(textNoN);

    // now we can build our secondary box
    var nBGText = "<tspan fill='black'></tspan><tspan fill='black'>" + n + "</tspan>  ",
      textN = addText(xBoxRight + 10, yText, nBGText, fontSizeText, fontColor);
    svgElement.appendChild(textN);
    var bbox = textN.getBBox(),
      xBoxN = xBoxRight + 5;
    svgElement.removeChild(textN);

    // now we want to build the full box with the n count set to transparent within the tspan to get proper width of box
    var text = addText(xText, yText, valueText, fontSizeText, fontColor);
    if (text.childNodes[1]) {
        var tspan = text.childNodes[1];
    } else {
        var tspan = text.childNodes[0];
    }
    tspan.innerHTML = "<tspan fill='transparent'>" + n + "</tspan>  ";

    svgElement.appendChild(text);
    var bbox = text.getBBox(),
      textWidth = bbox.width,
      textHeight = bbox.height,
      widthBox = textWidth * 1.2,
      heightBox = textHeight * 1.9,
      xBox = xText - (widthBox - textWidth)/2,
      yBox = yText - textHeight - 0.11*heightBox,
      rect = addRectangle(xBox, yBox, widthBox, heightBox, boxOutlineColor, boxFill, 2.5, boxId);
    svgElement.removeChild(text);

    // now we should create the background rectangle for our count n
    var n_value = addText(xBoxN, yBox, nBGText, fontSizeText, fontColor);
    svgElement.appendChild(n_value)
    var bbox = n_value.getBBox(),
      nWidth = bbox.width,
      nHeight = bbox.height,
      nBoxWidth = 1.1 * nWidth,
      nBoxHeight = 1.4 * nHeight,
      yBoxN = (yBox + heightBox/2) - (0.5 * nBoxHeight),
      rectN = addRectangle(xBoxN, yBoxN, nBoxWidth, nBoxHeight, "white", nBGColor, 1, "back");
    svgElement.removeChild(n_value);
    
    // add elements to SVG in proper order
    svgElement.appendChild(rect);
    svgElement.appendChild(rectN);
    svgElement.appendChild(text);
    svgElement.appendChild(textN);

    return svgElement;
  }

  /**
   * Adds a horizontal arrow pointing from box aId to box bId.
   * @param {SVGElement} svgElement SVG element you want to append an arrow to.
   * @param {String} aId Identifier for the first SVG rect you want to point from.
   * @param {String} bId Identifier for the second SVG rect you want to point to.
   * @param {Number} arrowHeadLength Length of arrow head.
   * @param {String} strokeColor 
   * @param {String} fillColor 
   * @returns Your updated SVG element with an arrow added.
   */
  function addSingleHorizontalArrow(svgElement, aId, bId, arrowHeadLength, strokeColor, fillColor) {
    // Currently assumes that aID is to the left of bID... would be easy to generalize with some logic
    var boxA = document.getElementById(aId),
      boxB = document.getElementById(bId),
      bboxA = boxA.getBBox(), // { x: Number, y: Number, width: Number, height: Number }
      bboxB = boxB.getBBox(),
      xA = bboxA.x,
      yA = bboxA.y,
      widthA = bboxA.width,
      heightA = bboxA.height,
      xB = bboxB.x,
      yB = bboxB.y,
      heightB = bboxB.height,
      firstOffset = 0.2 * heightA, // first offset in y from middle
      secondOffset = 0.2 * heightA; // second offset in y to create arrow head

    // find vertices of arrow
    var x0 = xA + widthA,
      y0 = yA + heightA/2 + firstOffset,
      x1 = xB - arrowHeadLength,
      y1 = y0,
      x2 = x1,
      y2 = y1 + secondOffset,
      x3 = xB,
      y3 = yB + heightB/2,
      x4 = x2,
      y4 = y2 - 2 * (secondOffset + firstOffset),
      x5 = x4,
      y5 = y4 + firstOffset,
      x6 = x0,
      y6 = y0 - 2 * firstOffset;

    var points = `${x0} ${y0}, ${x1} ${y1}, ${x2} ${y2}, ${x3} ${y3}, ${x4} ${y4}, ${x5} ${y5}, ${x6} ${y6}, ${x0} ${y0}`;

    var arrow = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    arrow.setAttribute('points', points);
    arrow.setAttribute('stroke', strokeColor);
    arrow.setAttribute('fill', fillColor);
    arrow.setAttribute('stroke-width', 1.5);

    svgElement.appendChild(arrow);

    return svgElement;
  }

  /**
   * Adds Bifurcating arrow from aId to bId/cId.
   * @param {SVGElement} svgElement SVG element you want to add arrow to.
   * @param {String} aId Identifier for box A.
   * @param {String} bId Identifier for box B.
   * @param {String} cId Identifier for box C.
   * @param {Number} arrowHeadLength 
   * @param {String} aStrokeColor 
   * @param {String} aFillColor 
   * @param {String} bStrokeColor 
   * @param {String} bFillColor 
   * @param {String} cStrokeColor 
   * @param {String} cFillColor 
   * @param {String} textLeft Text to be appended above the left branch of arrow.
   * @param {String} textRight Text to be appended above the right branch of arrow.
   * @returns Your SVG element with a bifurcating arrow appended.
   */
  function addBifurcatingArrow(svgElement, aId, bId, cId, arrowHeadLength, aStrokeColor, aFillColor, bStrokeColor, bFillColor, cStrokeColor, cFillColor, textLeft, textRight) {
    // aId is top box, bId is bottom left, cId is bottom right
    var boxA = document.getElementById(aId),
      boxB = document.getElementById(bId),
      boxC = document.getElementById(cId),
      bboxA = boxA.getBBox(), // { x: Number, y: Number, width: Number, height: Number }
      bboxB = boxB.getBBox(),
      bboxC = boxC.getBBox(), 
      xA = bboxA.x,
      yA = bboxA.y,
      widthA = bboxA.width,
      heightA = bboxA.height,
      xB = bboxB.x,
      yB = bboxB.y,
      widthB = bboxB.width,
      xC = bboxC.x,
      yC = bboxC.y,
      widthC = bboxC.width,
      widthOffset = 0.2 * heightA,
      tipOffset = 0.2 * heightA;

    // get vertices for first rectangle
    var v0x = xA + (widthA / 2) - widthOffset,
      v0y = yA + heightA + 1.1,
      v1x = v0x,
      v1y = (v0y + yB) / 2,
      v2x = v1x + (widthOffset * 2),
      v2y = v1y,
      v3x = v2x,
      v3y = v0y;

    var points = `${v0x} ${v0y}, ${v1x} ${v1y}, ${v2x} ${v2y}, ${v3x} ${v3y}`;

    var arrow = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    arrow.setAttribute('points', points);
    arrow.setAttribute('stroke', aStrokeColor);
    arrow.setAttribute('fill', aFillColor);
    arrow.setAttribute('stroke-width', 1.5);

    svgElement.appendChild(arrow);

    // get vertices for second rectangle
    var u0x = xA + (widthA / 2),
      u0y = v1y - widthOffset,
      u1x = xB + (widthB / 2) - widthOffset,
      u1y = u0y,
      u2x = u1x,
      u2y = yB - arrowHeadLength,
      u3x = u2x - tipOffset,
      u3y = u2y,
      u4x = xB + (widthB / 2),
      u4y = yB - 1,
      u5x = xB + (widthB / 2) + widthOffset + tipOffset,
      u5y = u2y,
      u6x = u5x - tipOffset,
      u6y = u5y,
      u7x = u6x,
      u7y = u0y + (2 * widthOffset),
      u8x = u0x,
      u8y = u7y;

    var points = `${u0x} ${u0y}, ${u1x} ${u1y}, ${u2x} ${u2y}, ${u3x} ${u3y}, ${u4x} ${u4y}, ${u5x} ${u5y}, ${u6x} ${u6y}, ${u7x} ${u7y}, ${u8x} ${u8y}`;

    var arrow = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    arrow.setAttribute('points', points);
    arrow.setAttribute('stroke', bStrokeColor);
    arrow.setAttribute('fill', bFillColor);
    arrow.setAttribute('stroke-width', 1.8);

    svgElement.appendChild(arrow);
    
    // get vertices for third rectangle
    var w0x = xA + (widthA / 2),
      w0y = v1y - widthOffset,
      w1x = xC + (widthC / 2) + widthOffset,
      w1y = w0y,
      w2x = w1x,
      w2y = yC - arrowHeadLength,
      w3x = w2x + tipOffset,
      w3y = w2y,
      w4x = xC + (widthC / 2),
      w4y = yC - 1,
      w5x = xC + (widthC / 2) - widthOffset - tipOffset,
      w5y = w2y,
      w6x = w5x + tipOffset,
      w6y = w5y,
      w7x = w6x,
      w7y = w0y + (2 * widthOffset),
      w8x = w0x,
      w8y = w7y;

    var points = `${w0x} ${w0y}, ${w1x} ${w1y}, ${w2x} ${w2y}, ${w3x} ${w3y}, ${w4x} ${w4y}, ${w5x} ${w5y}, ${w6x} ${w6y}, ${w7x} ${w7y}, ${w8x} ${w8y}`;

    var arrow = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    arrow.setAttribute('points', points);
    arrow.setAttribute('stroke', cStrokeColor);
    arrow.setAttribute('fill', cFillColor);
    arrow.setAttribute('stroke-width', 1.8);

    svgElement.appendChild(arrow);

    // add text elements
    var xTextL = (u0x + u1x) / 2,
      xTextR = (w1x + w0x) / 2,
      yText = u0y - widthOffset;

    var textL = addText(xTextL, yText, textLeft, 23, "white");
    svgElement.appendChild(textL);
    var bbox = textL.getBBox();
    svgElement.removeChild(textL);
    var textLWidth = bbox.width;
    xTextL -= textLWidth / 2;
    var textL = addText(xTextL, yText, textLeft, 23, "white");
    svgElement.appendChild(textL);
    textL.setAttribute("font-style", "italic");

    var textR = addText(xTextR, yText, textRight, 23, "white");
    svgElement.appendChild(textR);
    var bbox = textR.getBBox();
    svgElement.removeChild(textR);
    var textRWidth = bbox.width;
    xTextR -= textRWidth / 2;
    var textR = addText(xTextR, yText, textRight, 23, "white");
    svgElement.appendChild(textR);
    textR.setAttribute("font-style", "italic");

    return svgElement;
  }

  /**
   * Function for converting into a string with commas separating 3 digits.
   * @param {Number} x The number you want to convert.
   * @returns x with commas separating every 3 digits
   */
  function numberWithCommas(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  /**
   * Downloads an SVG with an id to path.
   * @param {String} SVGid Identifier for the SVG to download.
   * @param {String} DIVid Identifier for the div that contains the SVG.
   * @param {String} path Download path.
   */
  function downloadSVG(SVGid, DIVid, path) {
    var svg = document.getElementById(SVGid),
      rect = addRectangle(0, 0, 1450, 1000, "black", "black", "1px", "background"),
      svgSave = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgSave.setAttribute("id", "svgLeftSave")
    svgSave.setAttributeNS("http://www.w3.org/2000/xmlns/", "xmlns:xlink", "http://www.w3.org/1999/xlink");
    svgSave.appendChild(rect);
    svgSave.appendChild(svg);
    document.body.appendChild(svgSave);
    saveSVG(document.getElementById('svgLeftSave'), path);
    document.body.removeChild(svgSave);
    var div = document.getElementById(DIVid);
    div.appendChild(svg);
  }

  /**
   * Function that actually saves and downloads the SVG.
   * @param {SVGElement} svgEl SVG element you want to be saved.
   * @param {String} name Path for download
   */
  function saveSVG(svgEl, name) {
    svgEl.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    var svgData = svgEl.outerHTML;
    svgData = svgData.replaceAll('&nbsp;&nbsp;&nbsp;', '&#x20;');
    var preface = '<?xml version="1.0" standalone="no"?>\r\n',
      svgBlob = new Blob([preface, svgData], {type:"image/svg+xml;charset=utf-8"}),
      svgUrl = URL.createObjectURL(svgBlob),
      downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = name;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  }

})