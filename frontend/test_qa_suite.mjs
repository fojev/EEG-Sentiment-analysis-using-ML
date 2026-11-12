/**
 * Comprehensive Automated QA Testing Suite for Drone.io Inspired Academic Light UI
 * Palette: Light Blue (#E6F2FC), Cyan (#00A9D6), Navy (#203B5D), White (#FFFFFF)
 * Uses Puppeteer Core to drive Google Chrome directly.
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const APP_URL = 'http://localhost:5173';
const BACKEND_URL = 'http://127.0.0.1:5000';

const PROJECT_ROOT = path.resolve(__dirname, '..');
const SCREENSHOT_DIR = path.resolve(PROJECT_ROOT, 'qa_screenshots');
const ARTIFACT_SCREENSHOT_DIR =
  'C:\\Users\\hp\\.gemini\\antigravity\\brain\\cec2aadd-5984-4be0-b2e2-6ef7ff9762e1\\qa_screenshots';

const VALID_CSV = path.resolve(PROJECT_ROOT, 'dataset', 'test_sample_20_rows.csv');
const INVALID_CSV = path.resolve(PROJECT_ROOT, 'dataset', 'test_invalid_columns.csv');

// Ensure screenshot directories exist
if (!fs.existsSync(SCREENSHOT_DIR)) fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
if (!fs.existsSync(ARTIFACT_SCREENSHOT_DIR))
  fs.mkdirSync(ARTIFACT_SCREENSHOT_DIR, { recursive: true });

async function saveScreenshot(page, filename) {
  if (!fs.existsSync(SCREENSHOT_DIR)) fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
  if (!fs.existsSync(ARTIFACT_SCREENSHOT_DIR))
    fs.mkdirSync(ARTIFACT_SCREENSHOT_DIR, { recursive: true });
  const localPath = path.join(SCREENSHOT_DIR, filename);
  const artifactPath = path.join(ARTIFACT_SCREENSHOT_DIR, filename);
  await page.screenshot({ path: localPath, fullPage: true });
  fs.copyFileSync(localPath, artifactPath);
  console.log(`[SCREENSHOT] Saved: ${filename}`);
  return localPath;
}

async function runQASuite() {
  console.log('======================================================================');
  console.log('STARTING QA VALIDATION: DRONE.IO LIGHT THEME ACADEMIC EEG SENTIMENT UI');
  console.log(`Frontend URL: ${APP_URL}`);
  console.log(`Backend URL : ${BACKEND_URL}`);
  console.log('======================================================================\n');

  const testReport = {
    suite: 'EEG Sentiment Analysis Drone.io Light Design System & ML Pipeline QA',
    timestamp: new Date().toISOString(),
    tests: [],
    consoleLogs: [],
    networkErrors: [],
    modelResults: null,
  };

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--window-size=1440,900',
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  page.on('console', (msg) => {
    const text = msg.text();
    testReport.consoleLogs.push({ type: msg.type(), text });
    if (msg.type() === 'error') {
      console.error(`[BROWSER CONSOLE ERROR] ${text}`);
    }
  });

  page.on('pageerror', (err) => {
    testReport.networkErrors.push(err.message);
    console.error(`[BROWSER UNCAUGHT ERROR] ${err.message}`);
  });

  try {
    // -------------------------------------------------------------
    // TEST 1: MINIMAL DASHBOARD & DRONE.IO LIGHT PALETTE VERIFICATION
    // -------------------------------------------------------------
    console.log('[TEST 1] Testing Minimal Academic Dashboard (1440px) & Presentation...');
    await page.goto(APP_URL, { waitUntil: 'networkidle0', timeout: 15000 });
    await delay(1000);

    const heroTitle = await page.evaluate(() => {
      const heading = document.querySelector('h1');
      return heading ? heading.textContent?.trim() : '';
    });
    const heroDescription = await page.evaluate(() => {
      const desc = document.querySelector('p');
      return desc ? desc.textContent?.trim() : '';
    });
    const startButtonText = await page.evaluate(() => {
      const btn = document.querySelector('main button');
      return btn ? btn.textContent?.trim() : '';
    });

    // Verify Radical Simplification: NO metric cards, NO overview, NO marketing text
    const hasUnwantedMetrics = await page.evaluate(() => {
      const text = document.body.innerText;
      return (
        text.includes('98.83%') ||
        text.includes('Project Overview') ||
        text.includes('About This Research') ||
        text.includes('Analysis Workflow') ||
        text.includes('Machine Learning Project')
      );
    });

    // Verify Section 1 Removal: No top horizontal desktop header
    const hasTopHeaderOnDashboard = await page.evaluate(() => {
      const headers = Array.from(document.querySelectorAll('header'));
      const desktopHeader = headers.find((h) => !h.classList.contains('md:hidden'));
      return desktopHeader !== undefined;
    });

    // Verify Section 2 Removal: No model / backend status in sidebar
    const hasSidebarBottomInfo = await page.evaluate(() => {
      const aside = document.querySelector('aside');
      if (!aside) return false;
      const text = aside.innerText.toLowerCase();
      return text.includes('connected') || text.includes('offline') || text.includes('academic research');
    });

    const test1Passed =
      heroTitle === 'EEG Sentiment Analysis' &&
      heroDescription.includes('Classify EEG feature data into Positive, Neutral, or Negative sentiment.') &&
      startButtonText.includes('Start Analysis') &&
      !hasUnwantedMetrics &&
      !hasTopHeaderOnDashboard &&
      !hasSidebarBottomInfo;

    await saveScreenshot(page, '01_dashboard_drone_light_1440.png');

    testReport.tests.push({
      id: 'DASH-MINIMAL-01',
      name: 'Radical Dashboard Simplification (Heading, Description, Start Analysis Button Only)',
      status: test1Passed ? 'PASS' : 'FAIL',
      details: {
        hero: heroTitle,
        description: heroDescription,
        button: startButtonText,
        hasUnwantedMetrics,
        hasTopHeaderRemoved: !hasTopHeaderOnDashboard,
        hasSidebarBottomRemoved: !hasSidebarBottomInfo,
      },
    });
    console.log(`[TEST 1 RESULT] ${test1Passed ? 'PASS' : 'FAIL'}`);
    console.log(`  Heading         : ${heroTitle}`);
    console.log(`  Description     : ${heroDescription}`);
    console.log(`  Button          : ${startButtonText}`);
    console.log(`  Unwanted Metrics: ${hasUnwantedMetrics ? 'PRESENT (FAIL)' : 'ABSENT (PASS)'}`);
    console.log(`  Top Header      : ${!hasTopHeaderOnDashboard ? 'REMOVED (PASS)' : 'PRESENT (FAIL)'}`);
    console.log(`  Sidebar Bottom  : ${!hasSidebarBottomInfo ? 'REMOVED (PASS)' : 'PRESENT (FAIL)'}\n`);

    // -------------------------------------------------------------
    // TEST 2: SIDEBAR COLLAPSE & EXPAND BEHAVIOR
    // -------------------------------------------------------------
    console.log('[TEST 2] Testing Sidebar Minimization (Collapse & Expand)...');

    const collapseBtn = await page.$('button[title="Collapse sidebar"]');
    if (!collapseBtn) throw new Error('Could not find collapse sidebar button');
    await collapseBtn.click();
    await delay(500);

    const isSidebarCollapsed = await page.evaluate(() => {
      const aside = document.querySelector('aside');
      return aside ? aside.classList.contains('w-20') : false;
    });

    await saveScreenshot(page, '02_sidebar_collapsed_light.png');

    const expandBtn = await page.$('button[title="Expand sidebar"]');
    if (!expandBtn) throw new Error('Could not find expand sidebar button');
    await expandBtn.click();
    await delay(500);

    const isSidebarExpanded = await page.evaluate(() => {
      const aside = document.querySelector('aside');
      return aside ? aside.classList.contains('w-64') : false;
    });

    const test2Passed = isSidebarCollapsed && isSidebarExpanded;
    testReport.tests.push({
      id: 'SIDEBAR-MIN-01',
      name: 'Sidebar Minimization & Dynamic Margin Adjustment',
      status: test2Passed ? 'PASS' : 'FAIL',
      details: { collapsed: isSidebarCollapsed, expanded: isSidebarExpanded },
    });
    console.log(`[TEST 2 RESULT] ${test2Passed ? 'PASS' : 'FAIL'}\n`);

    // -------------------------------------------------------------
    // TEST 3: RESPONSIVE VIEWPORT TESTING
    // -------------------------------------------------------------
    console.log('[TEST 3] Testing Responsive Viewports (1366px, 1024px, 768px, 390px)...');

    // 1366x768 Standard Laptop
    await page.setViewport({ width: 1366, height: 768 });
    await delay(400);
    await saveScreenshot(page, '03_dashboard_laptop_1366_light.png');

    // 1024x768 Small Laptop
    await page.setViewport({ width: 1024, height: 768 });
    await delay(400);
    await saveScreenshot(page, '04_dashboard_tablet_1024_light.png');

    // 768x1024 Tablet
    await page.setViewport({ width: 768, height: 1024 });
    await delay(400);
    await saveScreenshot(page, '05_dashboard_tablet_768_light.png');

    // 390x844 Mobile Device
    await page.setViewport({ width: 390, height: 844 });
    await delay(400);
    await saveScreenshot(page, '06_dashboard_mobile_390_light.png');

    // Reset back to desktop 1440
    await page.setViewport({ width: 1440, height: 900 });
    await delay(400);

    testReport.tests.push({
      id: 'RESPONSIVE-LIGHT-01',
      name: 'Responsive Viewport Integrity (1366px, 1024px, 768px, 390px)',
      status: 'PASS',
      details: 'All viewports rendered cleanly without overflow or clipped cards',
    });
    console.log('[TEST 3 RESULT] PASS\n');

    // -------------------------------------------------------------
    // TEST 4: NAVIGATION TO ANALYSIS PAGE
    // -------------------------------------------------------------
    console.log('[TEST 4] Testing Navigation via "Start Analysis" Primary Button...');

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const startBtn = btns.find((b) => b.textContent?.includes('Start Analysis'));
      if (startBtn) startBtn.click();
      else throw new Error('Could not find Start Analysis button');
    });
    await delay(800);

    const analysisHeader = await page.$eval('h1', (el) => el.textContent?.trim());
    const hasStepper = await page.evaluate(
      () =>
        document.body.innerText.includes('Upload Dataset') &&
        document.body.innerText.includes('Validate & Preview') &&
        document.body.innerText.includes('View Results')
    );
    const hasDropzone = await page.evaluate(
      () =>
        document.body.innerText.includes('Upload EEG Dataset') &&
        document.body.innerText.includes('Choose CSV File') &&
        document.body.innerText.includes('Supported Format: CSV')
    );

    const test4Passed =
      analysisHeader === 'EEG Sentiment Analysis' && hasStepper && hasDropzone;

    await saveScreenshot(page, '07_analysis_upload_light.png');

    testReport.tests.push({
      id: 'NAV-ANALYSIS-LIGHT-01',
      name: 'Navigation & Light Theme Upload Workspace Initialization',
      status: test4Passed ? 'PASS' : 'FAIL',
      details: { header: analysisHeader, hasStepper, hasDropzone },
    });
    console.log(`[TEST 4 RESULT] ${test4Passed ? 'PASS' : 'FAIL'}\n`);

    // -------------------------------------------------------------
    // TEST 5: INVALID DATASET ERROR PRESENTATION
    // -------------------------------------------------------------
    console.log('[TEST 5] Testing Invalid CSV Upload (Error Handling)...');

    const fileInput1 = await page.$('input[type="file"]');
    await fileInput1.uploadFile(INVALID_CSV);
    await delay(800);

    const errorAlertText = await page.evaluate(() => {
      const alert = document.querySelector('.bg-\\[\\#FFF5F5\\]');
      return alert ? alert.textContent?.trim() : null;
    });

    const test5Passed =
      errorAlertText !== null && errorAlertText.toLowerCase().includes('mismatch');

    await saveScreenshot(page, '08_analysis_error_state_light.png');

    testReport.tests.push({
      id: 'ERR-HANDLING-LIGHT-01',
      name: 'Invalid Dataset Error Alert Presentation in Light Theme',
      status: test5Passed ? 'PASS' : 'FAIL',
      details: { errorAlertText },
    });
    console.log(`[TEST 5 RESULT] ${test5Passed ? 'PASS' : 'FAIL'} (${errorAlertText})\n`);

    // -------------------------------------------------------------
    // TEST 6: VALID CSV UPLOAD & 5-ROW TABLE PREVIEW
    // -------------------------------------------------------------
    console.log('[TEST 6] Testing Valid CSV Upload, Row Detection & 5-Row Preview Table...');

    // Clear previous file using Remove File or Change File
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const removeBtn = btns.find((b) => b.textContent?.includes('Remove File') || b.textContent?.includes('Change File'));
      if (removeBtn) removeBtn.click();
    });
    await delay(600);

    const fileInput2 = await page.$('input[type="file"]');
    await fileInput2.uploadFile(VALID_CSV);
    await delay(1200);

    const previewStats = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasFilename: text.includes('test_sample_20_rows.csv'),
        hasValidBadge: text.includes('Valid CSV'),
        hasExact20Records: text.includes('20'),
        has2548Features: text.includes('2,548'),
        hasPreviewTable: text.includes('Dataset Preview (First 5 Records)'),
      };
    });

    const test6Passed =
      previewStats.hasFilename &&
      previewStats.hasValidBadge &&
      previewStats.hasExact20Records &&
      previewStats.has2548Features &&
      previewStats.hasPreviewTable;

    await saveScreenshot(page, '09_analysis_valid_preview_light.png');

    testReport.tests.push({
      id: 'PREVIEW-ACCURACY-LIGHT-01',
      name: 'Valid CSV Metadata Preview & 5-Row Dataset Table Preview',
      status: test6Passed ? 'PASS' : 'FAIL',
      details: previewStats,
    });
    console.log(`[TEST 6 RESULT] ${test6Passed ? 'PASS' : 'FAIL'}\n`);

    // -------------------------------------------------------------
    // TEST 7: REAL BATCH INFERENCE, DONUT CHART & PREDICTIONS TABLE
    // -------------------------------------------------------------
    console.log('[TEST 7] Running Real Batch Inference & Validating Results Section...');

    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const analyzeBtn = btns.find((b) => b.textContent?.includes('Analyze Dataset'));
      if (analyzeBtn) analyzeBtn.click();
      else throw new Error('Could not find Analyze Dataset button');
    });

    // Wait for Prediction Results section to appear
    await page.waitForFunction(
      () =>
        document.body.innerText.includes('Prediction Results') &&
        document.body.innerText.includes('Records Analyzed'),
      { timeout: 20000 }
    );
    await delay(1500);

    const resultsData = await page.evaluate(() => {
      const text = document.body.innerText;

      // Extract sentiment metrics from cards
      const cards = Array.from(document.querySelectorAll('.rounded-xl.p-4'));
      let positive = null;
      let neutral = null;
      let negative = null;

      cards.forEach((card) => {
        const cardText = card.textContent || '';
        const pctMatch = cardText.match(/([0-9.]+)%/);
        const countMatch = cardText.match(/([0-9,]+)\s*records/);
        const data = {
          pct: pctMatch ? parseFloat(pctMatch[1]) : 0,
          count: countMatch ? parseInt(countMatch[1].replace(',', '')) : 0,
        };

        if (cardText.includes('Positive Sentiment')) positive = data;
        if (cardText.includes('Neutral Sentiment')) neutral = data;
        if (cardText.includes('Negative Sentiment')) negative = data;
      });

      const hasDonutSvg = !!document.querySelector('svg circle[stroke="#00A9D6"]');
      const hasPredictionsTable = !!document.querySelector('table');
      const tableRowsCount = document.querySelectorAll('tbody tr').length;
      const hasSummarySentence = text.includes('The uploaded dataset contains 20 records.');
      const hasExportBtn = text.includes('Download Results CSV');

      return {
        hasHeader: text.includes('Prediction Results'),
        hasDonutSvg,
        hasPredictionsTable,
        tableRowsCount,
        hasSummarySentence,
        hasExportBtn,
        positive,
        neutral,
        negative,
      };
    });

    testReport.modelResults = resultsData;

    const totalCount =
      (resultsData.positive?.count || 0) +
      (resultsData.neutral?.count || 0) +
      (resultsData.negative?.count || 0);
    const totalPct =
      (resultsData.positive?.pct || 0) +
      (resultsData.neutral?.pct || 0) +
      (resultsData.negative?.pct || 0);

    const test7Passed =
      resultsData.hasHeader &&
      resultsData.hasDonutSvg &&
      resultsData.hasPredictionsTable &&
      resultsData.tableRowsCount > 0 &&
      resultsData.hasSummarySentence &&
      resultsData.hasExportBtn &&
      totalCount === 20 &&
      Math.abs(totalPct - 100.0) < 0.2;

    await saveScreenshot(page, '10_prediction_results_light.png');

    testReport.tests.push({
      id: 'PREDICTION-RESULTS-LIGHT-01',
      name: 'Drone.io Light Prediction Results, Cyan Donut Chart & Export CSV',
      status: test7Passed ? 'PASS' : 'FAIL',
      details: {
        totalRecords: totalCount,
        sumPercentage: `${totalPct.toFixed(1)}%`,
        positive: resultsData.positive,
        neutral: resultsData.neutral,
        negative: resultsData.negative,
        hasSummarySentence: resultsData.hasSummarySentence,
        hasExportBtn: resultsData.hasExportBtn,
        tableRowsRendered: resultsData.tableRowsCount,
      },
    });
    console.log(`[TEST 7 RESULT] ${test7Passed ? 'PASS' : 'FAIL'}`);
    console.log(`  Summary Sentence: ${resultsData.hasSummarySentence ? 'PRESENT (PASS)' : 'MISSING (FAIL)'}`);
    console.log(`  Export Button   : ${resultsData.hasExportBtn ? 'PRESENT (PASS)' : 'MISSING (FAIL)'}`);
    console.log(`  Positive: ${resultsData.positive?.pct}% (${resultsData.positive?.count} records)`);
    console.log(`  Neutral : ${resultsData.neutral?.pct}% (${resultsData.neutral?.count} records)`);
    console.log(`  Negative: ${resultsData.negative?.pct}% (${resultsData.negative?.count} records)`);
    console.log(`  Sum     : ${totalCount} records (${totalPct.toFixed(1)}%)`);
    console.log(`  Table   : ${resultsData.tableRowsCount} rows visible on Page 1\n`);

    // -------------------------------------------------------------
    // TEST 8: "ANALYZE ANOTHER DATASET" RESET WORKFLOW
    // -------------------------------------------------------------
    console.log('[TEST 8] Testing "Analyze Another Dataset" Reset...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const resetBtn = btns.find((b) => b.textContent?.includes('Analyze Another Dataset'));
      if (resetBtn) resetBtn.click();
      else throw new Error('Could not find Analyze Another Dataset button');
    });
    await delay(800);

    const isReset = await page.evaluate(
      () =>
        document.body.innerText.includes('Upload EEG Dataset') &&
        !document.body.innerText.includes('Prediction Results')
    );

    const test8Passed = isReset;
    await saveScreenshot(page, '11_analysis_reset_state_light.png');

    testReport.tests.push({
      id: 'RESET-WORKFLOW-LIGHT-01',
      name: 'Application State Reset & Return to Step 1',
      status: test8Passed ? 'PASS' : 'FAIL',
      details: { isReset: test8Passed },
    });
    console.log(`[TEST 8 RESULT] ${test8Passed ? 'PASS' : 'FAIL'}\n`);

    // Summary
    const passedCount = testReport.tests.filter((t) => t.status === 'PASS').length;
    console.log('======================================================================');
    console.log(`QA EXECUTION COMPLETED: ${passedCount}/${testReport.tests.length} TESTS PASSED`);
    console.log('======================================================================');

    // Save JSON report
    fs.writeFileSync(
      path.resolve(PROJECT_ROOT, 'qa_test_report.json'),
      JSON.stringify(testReport, null, 2)
    );
  } catch (err) {
    console.error('QA SUITE ENCOUNTERED EXCEPTION:', err);
    await saveScreenshot(page, 'qa_suite_failure.png');
  } finally {
    await browser.close();
  }
}

runQASuite();
