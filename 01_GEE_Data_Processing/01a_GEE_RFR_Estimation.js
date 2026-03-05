// /**
// * @title Remote Sensing-based Forest Litterfall (PFL) Estimation (Bug Fixed)
// * @description 
// * 本脚本为全球/半球尺度森林枯落物(PFL)估算的核心反演代码。
// * 集成了多源 MODIS 产品协调、严格 QA 位掩膜去云去雪、历史极值动态计算。
// * 采用受生物群落约束的自适应空间邻域插值 (Biome-constrained Adaptive Focal Mean)。
// * 内置 70/30 独立验证集测试，客观评估估算精度 (R2 & RMSE)。
// * * Target Biome: ENF (Evergreen Needleleaf Forests, IGBP Class 1)
// * Target Year: 2020
// */

// ==========================================
// 1. 核心参数与研究区设定 (Spatiotemporal Setup)
// ==========================================
var target_year = 2020;
var target_class = 1; 

var geometry = ee.Geometry.Rectangle([110, 20, 120, 30]);
Map.centerObject(geometry, 6);

// ==========================================
// 2. 遥感数据源与资产导入
// ==========================================
var mcd12q1 = ee.ImageCollection("MODIS/061/MCD12Q1");
var mod09a1 = ee.ImageCollection("MODIS/061/MOD09A1");
var mod13q1 = ee.ImageCollection("MODIS/061/MOD13Q1"); 
var mod15a2h = ee.ImageCollection("MODIS/061/MOD15A2H");
var mod17a3hgf = ee.ImageCollection("MODIS/061/MOD17A3HGF");

// 确保资产路径正确，将 ML_sample.xlsx 文件的 RFR 表格上传替换
var trainingFeature = ee.FeatureCollection("./RFR");

// ==========================================
// 3. 严格的质量控制 (QA Masking)
// ==========================================
var getQABits = function(image, start, end, newName) {
    var pattern = 0;
    for (var i = start; i <= end; i++) { pattern += Math.pow(2, i); }
    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start);
};

function maskMOD09(image) {
    var qa = image.select('StateQA');
    var mask = getQABits(qa, 0, 1, 'cloud').eq(0)
        .and(getQABits(qa, 2, 2, 'shadow').eq(0))
        .and(getQABits(qa, 12, 12, 'snow').eq(0));
    return image.updateMask(mask);
}

function maskMOD15(image) {
    var qa = image.select('FparLai_QC');
    return image.updateMask(getQABits(qa, 0, 0, 'qc').eq(0).and(getQABits(qa, 5, 7, 'c').eq(0)));
}

function maskMOD13(image) {
    var qa = image.select('DetailedQA');
    return image.updateMask(getQABits(qa, 0, 1, 'q').lte(1).and(getQABits(qa, 14, 14, 's').eq(0)));
}

// ==========================================
// 4. 空间空缺填补 (Optimized Gap-Filling)
// ==========================================
function calmean(image, bandName) {
    return ee.Number(image.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geometry,
        scale: 5000, 
        maxPixels: 1e13,
        tileScale: 16 
    }).get(bandName));
}

var isfrt = mcd12q1.filterDate(target_year + '-01-01', target_year + '-12-31')
                   .first().select('LC_Type1').eq(target_class).selfMask();

function fillGaps(image, forestMask, fallbackMean) {
    var maskedImage = image.updateMask(forestMask);
    var regionalMean = maskedImage.focal_mean({radius: 10000, units: 'meters', iterations: 1});
    return maskedImage.unmask(regionalMean).unmask(fallbackMean).updateMask(forestMask);
}

// ==========================================
// 5. 特征工程 (Biophysical Indices)
// ==========================================
function createMOD(mst_date, med_date) {
    var filtered = mod15a2h.filterDate(mst_date, med_date).filterBounds(geometry).map(maskMOD15).mean();
    var lai = filtered.select('Lai_500m').multiply(0.1).clamp(0, 10).rename('LAI');
    var fpar = filtered.select('Fpar_500m').multiply(0.01).clamp(0, 1).rename('FPAR');
    return ee.Image([fillGaps(lai, isfrt, calmean(lai, 'LAI')), 
                     fillGaps(fpar, isfrt, calmean(fpar, 'FPAR'))]);
}

function createNPP(year) {
    var npp = mod17a3hgf.filterDate(year + '-01-01', year + '-12-31').select('Npp').mean().multiply(0.0001).rename('NPP');
    return fillGaps(npp, isfrt, calmean(npp, 'NPP'));
}

function createSSI(mst_date, med_date) {
    var img = mod09a1.filterDate(mst_date, med_date).filterBounds(geometry).map(maskMOD09).mean().multiply(0.0001);
    var nir = img.select('sur_refl_b02').clamp(0, 1);
    var red = img.select('sur_refl_b01').clamp(0, 1);
    var green = img.select('sur_refl_b04').clamp(0, 1);
    var blue = img.select('sur_refl_b03').clamp(0, 1);
    var ssi = nir.subtract(red).divide(214).subtract(green.subtract(blue).divide(86).max(0))
                 .divide(nir.subtract(red).divide(214).add(green.subtract(blue).divide(86).max(0)))
                 .rename('SSI').clamp(0, 1);
    return fillGaps(ssi, isfrt, calmean(ssi, 'SSI'));
}

function createVCI(vst_date, ved_date) {
    var eviCol = mod13q1.filterDate(ee.Date(vst_date).advance(-5, 'year'), ee.Date(ved_date).advance(5, 'year'))
                        .map(maskMOD13).select('EVI').map(function(i){return i.multiply(0.0001)});
    var eviMinMax = eviCol.reduce(ee.Reducer.minMax());
    var currentEVI = mod13q1.filterDate(vst_date, ved_date).map(maskMOD13).select('EVI').mean().multiply(0.0001);
    var vci = currentEVI.subtract(eviMinMax.select('EVI_min')).divide(eviMinMax.select('EVI_max').subtract(eviMinMax.select('EVI_min'))).rename('VCI').clamp(0, 1);
    return fillGaps(vci, isfrt, calmean(vci, 'VCI'));
}

// ==========================================
// 6. 机器学习建模与向量化精度评估
// ==========================================
var bands = ['FPAR', 'LAI', 'NPP', 'SSI', 'VCI'];
var target_prop = 'PFL';

var cleanData = trainingFeature.filter(ee.Filter.notNull(bands.concat([target_prop])));
var withRandom = cleanData.randomColumn('random', 42); 
var trainData = withRandom.filter(ee.Filter.lt('random', 0.7));
var testData = withRandom.filter(ee.Filter.gte('random', 0.7));

var classifier = ee.Classifier.smileRandomForest({
        numberOfTrees: 500,    
        minLeafPopulation: 5, 
        seed: 42               
    }).setOutputMode('REGRESSION')
    .train({ features: trainData, classProperty: target_prop, inputProperties: bands });

/**
 * 核心修复：向量化聚合函数
 * 避免了 Computed value too large 错误
 */
function evaluateModelMetrics(evalData, classCol, predCol, prefix) {
    var classified = evalData.classify(classifier, predCol);
    
    // 计算观测值均值
    var meanActual = ee.Number(classified.aggregate_mean(classCol));
    
    // 计算 SS_res (残差平方和)
    var ssRes = ee.Number(classified.map(function(f) {
        var diff = ee.Number(f.get(classCol)).subtract(ee.Number(f.get(predCol)));
        return f.set('resSq', diff.pow(2));
    }).aggregate_sum('resSq'));
    
    // 计算 SS_tot (总离差平方和)
    var ssTot = ee.Number(classified.map(function(f) {
        var diff = ee.Number(f.get(classCol)).subtract(meanActual);
        return f.set('totSq', diff.pow(2));
    }).aggregate_sum('totSq'));
    
    var r2 = ee.Number(1).subtract(ssRes.divide(ssTot));
    var rmse = ssRes.divide(evalData.size()).sqrt();
    
    // print(prefix + ' 样本数:', classified.size());
    print(prefix + ' R^2:', r2);
    print(prefix + ' RMSE:', rmse);

}

print('=== 机器学习验证评估 (Optimized) ===');
evaluateModelMetrics(trainData, target_prop, 'predicted_PFL', 'Training');
evaluateModelMetrics(testData, target_prop, 'predicted_PFL', 'Validation');

// ==========================================
// 7. 逐月预测与导出 (Production Stream)
// ==========================================
var spatialClassifier = ee.Classifier.smileRandomForest({
        numberOfTrees: 500,    
        minLeafPopulation: 5,  
        seed: 42               
    }).setOutputMode('REGRESSION')
    .train({ features: cleanData, classProperty: target_prop, inputProperties: bands });

var npp_yearly = createNPP(target_year);

for (var m = 8; m <= 8; m++) {
    var mst_date = ee.Date.fromYMD(target_year, m, 1);
    var med_date = mst_date.advance(1, 'month');
    
    var mod_lf = createMOD(mst_date, med_date);
    var ssi_monthly = createSSI(mst_date, med_date);
    var vci_monthly = createVCI(mst_date, med_date);
    
    var inStack = ee.Image([
        mod_lf.select('FPAR'), mod_lf.select('LAI'), 
        npp_yearly, ssi_monthly, vci_monthly
    ]).updateMask(isfrt).clip(geometry).toFloat();
    
    var predictedPFL = inStack.classify(spatialClassifier, 'PFL_Density');
    var out_name = 'PFL_ENF_2020_' + m;
    
    Export.image.toDrive({
        image: predictedPFL,
        description: out_name,
        fileNamePrefix: out_name,
        folder: 'Global_PFL',
        region: geometry,
        scale: 500,
        crs: "EPSG:4326",
        maxPixels: 1e13
    });
}


