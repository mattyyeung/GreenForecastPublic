
const icons = {
  clear: 'sunny',
  cloudy: 'cloudy',
  dusty: 'windy',
  fog: 'fog',
  frost: 'frost',
  hazy: 'hazy',
  heavy_shower: 'rain',
  light_rain: 'rain',
  light_shower: 'shower',
  mostly_sunny: 'partly_cloudy',
  partly_cloudy: 'partly_cloudy',
  rain: 'rain',
  shower: 'shower',
  snow: 'frost',
  storm: 'storm',
  sunny: 'sunny',
  tropical_cyclone: 'storm',
  windy: 'windy',
  '': 'partly_cloudy',
}

const TIMEZONES = {
  'NSW1': 'Australia/Sydney',
  'QLD1': 'Australia/Brisbane',
  'SA1': 'Australia/Adelaide',
  'TAS1': 'Australia/Sydney',
  'VIC1': 'Australia/Sydney',
}

const CITIES = {
  'NSW1': 'Sydney',
  'QLD1': 'Brisbane',
  'SA1': 'Adelaide',
  'TAS1': 'Hobart',
  'VIC1': 'Melbourne',
}

const REGION_VERBOSE = {
  'NSW1': 'NSW',
  'QLD1': 'Queensland',
  'SA1': 'South Australia',
  'TAS1': 'Tasmania',
  'VIC1': 'Victoria',
}

const ONE_HOUR = 60 * 60 * 1000;

// Y-limits for charts
const PRICE_MAX = 100;
const PRICE_MIN = -100;
const GREENNESS_MAX = 100;
const GREENNESS_MIN = 0;

// global holding latest forecast data
let latestForecasts;

// getRegion() returns the region code currently selected
function getRegion() { return document.querySelector('input[name="region"]:checked').value; }

// applyTimezoneToUtc() takes a time in UTC and a target timezone, returns the clock time in that timezone at the same instant as the original time. 
// TODO: if anything more complicated than this is required, just use a library. 
function applyTimezoneToUtc(utc, timezone) {
  // Write a string with the utc time in the new timezone.
  // Use Sweden's locale string because it is similar to isoformat (no standard way of writing isoformat with in a different timezone)
  let dateString = utc.toLocaleString('sv-SE', {timeZone: timezone});
  // Parse that datestring as if it's UTC (even though it's not). Use D3 because safari doesn't parse %Y-%m-%d %H:%M:%S with Date.parse()
  const parseTime = d3.timeParse("%Y-%m-%d %H:%M:%S");
  // Now parse that string
  return parseTime(dateString);
}

// formatHoursAmpm() takes a time object and outputs a string with format '4pm', note lower case and no leading 0.
function formatHoursAmpm(time) {
  const hours = Number(d3.timeFormat('%I')(time)); // Hack to remove leading 0 from 12 hour time, eg want 5 not 05pm
  const ampm = d3.timeFormat('%p')(time).toLowerCase();
  return hours + ampm;
}

// render() (re)draws the entire table. Called whenever a new region is selected.
function render() {
  const region = getRegion();
  const forecasts = latestForecasts;

  // save current region in localstorage
  localStorage.setItem('region', region);

  // baseTimeInRegionTimezone is time when the forecast was made. We need to pad this many hours from midnight.
  const baseTimeUtc = new Date(Date.parse(forecasts.baseTimeUtc));
  const baseTimeInRegionTimezone = applyTimezoneToUtc(baseTimeUtc, TIMEZONES[region]);
  // nowInRegionTimezone current time in the region's timezone. Will often be a little after baseTimeInRegionTimezone but could be a fair bit after if forecasts aren't updated. 
  const nowInRegionTimezone = applyTimezoneToUtc(new Date(), TIMEZONES[region]);
  const startOfPastDataUtc = new Date(Date.parse(forecasts.pastTimestampsUtc[0]));
  const startOfPastDataLocal = applyTimezoneToUtc(startOfPastDataUtc, TIMEZONES[region]); // time of earliest past datapoint in local region timezone.

  let today = new Date(Date.parse(forecasts.baseTimeNem)); // use baseTimeNem because that's what weather data assumes when lablling 'today'.
  today.setHours(0);
  today.setMinutes(0); // SA timezone has non-integer hours!!

  renderSnapshot(region, forecasts);

  // Days of week
  d3.selectAll('.day-header').data(forecasts.weather.dayLabels.slice(1)).text(d => d); // slice(1) because skipping first day
  d3.selectAll('.day-label-bottom').data(forecasts.weather.dayLabels.slice(1)).text(d => d); // slice(1) because skipping first day

  // Max greenness
  const maxGreenness = forecasts.dayMaxs[region + '_Greenness'].slice(1); // skip first item because only displaying 6, not 7 days. 
  d3.selectAll('.max-greenness').data(maxGreenness).text(d => `${Math.round(d.max)}`);
  d3.selectAll('.max-greenness-time').data(maxGreenness).text(d => {
    const utc = new Date(Date.parse(d.utc));
    const local = applyTimezoneToUtc(utc, TIMEZONES[region]);
    return `${formatHoursAmpm(local)}`;
  });

  // Weather heading 
  d3.select('.cityname').text(CITIES[region]);

  // Weather
  d3.selectAll('.temperature')
    .data(forecasts.weather[region].map(day => day.max_temp).slice(1))  // slice(1) because skipping first day
    .text(d => 'Max: ' + d + '°C');

  d3.selectAll('.weather-icon')
    .data(forecasts.weather[region].map(day => icons[day.icon] + '.svg').slice(1))  // slice(1) because skipping first day
    .attr('src', d => d);

  // Test: - worst case data
  // forecasts.forecasts.NSW1_Price[0] = -1000;
  // forecasts.forecasts.NSW1_Price[1] = 1000;
  // forecasts.forecasts.NSW1_Greenness[0] = 100;
  // forecasts.forecasts.NSW1_Greenness[1] = 0;

  // Draw charts
  const forecastGreenness = forecasts.forecasts[region + '_Greenness'];
  const pastGreenness = forecasts.past[region + '_Greenness'];
  greennessChart.draw(nowInRegionTimezone, baseTimeInRegionTimezone, startOfPastDataLocal, today, pastGreenness, forecastGreenness, forecasts.highestEverGreenness[region].value);

  const forecastFuel = forecasts.forecastGenByFuel[region];
  const pastFuel = forecasts.pastGenByFuel[region];
  fuelChart.draw(nowInRegionTimezone, baseTimeInRegionTimezone, startOfPastDataLocal, today, pastFuel, forecastFuel, null);

  const forecastPriceInCentsPerKWh = forecasts.forecasts[region + '_Price'].map(x => x / 10);
  const pastPriceInCentsPerKWh = forecasts.past[region + '_Price'].map(x => x / 10);
  priceChart.draw(nowInRegionTimezone, baseTimeInRegionTimezone, startOfPastDataLocal, today, pastPriceInCentsPerKWh, forecastPriceInCentsPerKWh, null);

  resetChartTooltipToNow();

  // last-updated
  const hours = formatHoursAmpm(baseTimeInRegionTimezone);
  const date = d3.timeFormat(' %a %d %b ')(baseTimeInRegionTimezone);
  d3.select('#last-updated').text(() => "Last Updated: " + hours + date + CITIES[region] + " time");
}

// renderSnapshot() updates all the data in the snapshot (top) section
function renderSnapshot(region, forecasts) {

  // current greenness tile
  document.querySelectorAll('.region-name').forEach(e => {
    e.innerHTML = REGION_VERBOSE[region];
  });
  const currentGreenness = forecasts.past[region + '_Greenness'].slice(-1)[0];
  document.getElementById('current-greenness').innerHTML = Math.round(currentGreenness);
  const baseTimeUtc = new Date(Date.parse(forecasts.baseTimeUtc));
  const baseTimeInRegionTimezone = applyTimezoneToUtc(baseTimeUtc, TIMEZONES[region]);
  const timeString = `${formatHoursAmpm(baseTimeInRegionTimezone)} ${d3.timeFormat('%A')(baseTimeInRegionTimezone)}`
  document.getElementById('current-greenness-time').innerHTML = timeString;


  // today / tomorrow  max/min
  document.getElementById('today-max').innerHTML = Math.round(forecasts.dayMaxs[region + '_Greenness'][7].max);
  document.getElementById('today-min').innerHTML = Math.round(forecasts.dayMins[region + '_Greenness'][7].min);
  document.getElementById('tomorrow-max').innerHTML = Math.round(forecasts.dayMaxs[region + '_Greenness'][8].max);
  document.getElementById('tomorrow-min').innerHTML = Math.round(forecasts.dayMins[region + '_Greenness'][8].min);

  // recommendations
  document.getElementById('ev-charge-time').innerHTML = forecasts.recommendations.ev[region];
  document.getElementById('reduce-usage').innerHTML = forecasts.recommendations.reduceUsage[region];
  document.getElementById('highest-ever-greenness').innerHTML = forecasts.highestEverGreenness[region].value.toString() + "%";
  const highestEverDate = applyTimezoneToUtc(new Date(Date.parse(forecasts.highestEverGreenness[region].utc)), TIMEZONES[region]);
  document.getElementById('highest-ever-greenness-date').innerHTML = d3.timeFormat('%e %b %Y')(highestEverDate);

}

class Chart {

  constructor(chartId) {
    this.id = chartId;
    this.isGreenness = this.id === '#greenness-chart';
    this.isFuel = this.id === '#fuel-chart';
    this.isPrice = this.id === '#price-chart';
  }

  // draw() replaces any existing svg with a new one using data
  draw(now, baseTimeLocal, startOfPastDataLocal, today, pastData, forecastData, highestEverGreenness) {
    this.now = now;
    this.baseTimeLocal = baseTimeLocal;
    this.today = today;

    const chartHeight = 250;
    const chartMarginTop = 4; // vertical margin above chart
    const chartMarginBottom = 20; // vertical margin below chart
    const scaleWidth = 40;
    // format string for the y data
    const customFormat = this.isPrice ? s => d3.format('d')(s) + '¢' : s => d3.format('d')(s) + '%';

    // get full width of chart svg
    // offsetwidth gets rounded width but for some reason the flexbox seem to be using floor instead, making this 1px off sometimes
    // let width = document.querySelector(this.id).offsetWidth;
    // TODO: this still doesn't work, because width of the svg is 100%. So sometimes it's just 1px off. 
    let width = Math.floor(document.querySelector(this.id).getBoundingClientRect().width); // * 1.8571428;

    // set up the y scale (static) svg
    d3.select(this.id + '-scale').select('.chart-scale').remove();
    let yScale = d3.select(this.id + '-scale')
      .append('svg')
      .attr('width', scaleWidth)
      .attr('height', chartHeight + chartMarginTop + chartMarginBottom)
      .attr('class', 'chart-scale')
      .append('g')
        .attr('transform', 'translate(' + (scaleWidth-1)  + ',' + chartMarginTop + ')'); //+ document.querySelector(this.id + '-scale').offsetWidth + '

    // Make the svg
    d3.select(this.id).select('svg').remove();
    let svg = d3.select(this.id)
      .append('svg')
      .attr('class', 'chart')
      .attr('width', '100%')
      .attr('height', chartHeight + chartMarginTop + chartMarginBottom)
      .attr('preserveAspectRatio','none')
      .attr('viewBox', '0 0 ' + width + ' ' + (chartHeight + chartMarginTop + chartMarginBottom))
      .append('g')
        .attr('transform', 'translate(-1,' + chartMarginTop + ')');

    // let x = d3.scaleLinear().range([0, width]);
    const xRange = [0, width-1]
    this.x = d3.scaleTime().range(xRange);
    this.y = d3.scaleLinear().range([chartHeight, 0]);

    // this.x.domain([today, new Date(today.getTime() + 7 * 24 * ONE_HOUR)]);
    const minX = new Date(today.getTime() - 6 * 24 * ONE_HOUR);
    const maxX = new Date(today.getTime() + 7 * 24 * ONE_HOUR);
    this.x.domain([minX, maxX]);

    // map x values; offset because forecasts don't necessarily start at midnight.
    // first forecast is 2hrs after baseTimeLocal, ie (i+2)
    const forecastXCount = this.isFuel ? forecastData.Wind.length : forecastData.length
    const forecastXValues = [...Array(forecastXCount)].map((_, i) => new Date(baseTimeLocal.getTime() + (i+2) * ONE_HOUR));
    // x values for past times. Reference off baseTimeLocal, which is in the region's timezone. The last datapoint of the history is at baseTimeLocal.
    const pastXCount = this.isFuel ? pastData.Wind.length : pastData.length
    const pastXValues = [...Array(pastXCount)].map((_, i) => new Date(startOfPastDataLocal.getTime() + i * ONE_HOUR));


    // Y-axis domain
    if (this.isPrice) {
      // for price: set domain according to min/max of data
      let minY = d3.min(forecastData.concat(pastData)); 
      let maxY = d3.max(forecastData.concat(pastData)); 
      minY = Math.min(0, minY);  // y-min can't ever be greater than 0 (but below 0 is ok for negative prices)
      minY = Math.max(PRICE_MIN, minY);  // clamp
      maxY = Math.min(PRICE_MAX, maxY);  // clamp
      this.y.domain([minY, maxY]).nice().clamp(true);
    } else {
      // for greenness and fuel
      this.y.domain([GREENNESS_MIN, GREENNESS_MAX]).nice().clamp(true);
    }

    if (this.isFuel) {
      this.drawFuelChart(svg, pastData, forecastData, pastXCount, forecastXCount, pastXValues, forecastXValues);
    }


    // x axis
    svg.append('g')
      .attr('class', 'axis')
      .attr('transform', 'translate(0,' + chartHeight + ')')
      .call(d3.axisBottom().scale(this.x));

    // vertical gridlines
    svg.append('g')
      .attr('class', 'gridlines')
      .call(d3.axisBottom().scale(this.x).ticks(12).tickSize(chartHeight).tickFormat(''))

    // y axis
    svg.append('g')
      .attr('class', 'axis')
      // .attr('transform', 'translate(' + 0 + ',0)')
      .call(d3.axisLeft().scale(this.y));

    // Draw Y scale in the small static svg (for the scale only)
    yScale.append('g')
      .attr('class', 'axis')
      // .attr('transform', 'translate(' + 0 + ',0)')
      .call(d3.axisLeft().scale(this.y).ticks(4).tickFormat(customFormat));

    // horizontal gridlines
    svg.append('g')
      .attr('class', 'gridlines')
      .call(d3.axisLeft().scale(this.y).ticks(4).tickSize(-1 * width).tickFormat(''))

    // vertical line for current time
    svg.append('line')
      .attr('class', 'now-line')
      .attr('x1', this.x(now))
      .attr('y1', 0)
      .attr('x2', this.x(now))
      .attr('y2', chartHeight);

    // Draw horizontal 'highest ever greenness' line
    if (this.isGreenness && highestEverGreenness < 100) {
      svg.append('path')
        .datum([highestEverGreenness, highestEverGreenness])
        .attr('class', 'highest-ever-greenness-line')
        .attr('d', d3.line()
          .x((_,i) => xRange[i])
          .y(d => this.y(d))
          );

      svg.append("text")
        .attr('class', 'highest-ever-greenness-text')
        .attr("x", xRange[1] - 100)
        .attr("y", this.y( highestEverGreenness+2))
        .text((d) => "Record high");
      svg.append("text")
        .attr('class', 'highest-ever-greenness-text')
        .attr("x", xRange[0] + 503)
        .attr("y", this.y( highestEverGreenness+2))
        .text((d) => "Record high");
    }

    if (this.isPrice) {
      svg.append('path')
        .datum([0, 0])
        .attr('class', 'zero-price-line')
        .attr('d', d3.line()
          .x((_,i) => xRange[i])
          .y(d => this.y(d))
          );
      }

    if (this.isGreenness || this.isPrice) {
      // Draw path for the forecast data - twice so can overlay dotted line
      svg.append('path')
        .datum(forecastData)
        .attr('class', 'chart-line')
        .attr('d', d3.line()
          .x((_,i) => this.x(forecastXValues[i]))
          .y(d => this.y(d))
          );
      svg.append('path')
        .datum(forecastData)
        .attr('class', 'chart-line-forecast')
        .attr('d', d3.line()
          .x((_,i) => this.x(forecastXValues[i]))
          .y(d => this.y(d))
          );

      // Draw path for past data 
      svg.append('path')
        .datum(pastData)
        .attr('class', 'chart-line')
        .attr('d', d3.line()
          .x((_,i) => this.x(pastXValues[i]))
          .y(d => this.y(d))
          );
    }

    // tooltips
    // create vertical line for mouse hover
    svg.append('line')
      .attr('class', 'hover-line')
      .attr('x1', this.x(now))
      .attr('y1', -1 * chartMarginTop)
      .attr('x2', this.x(now))
      .attr('y2', chartHeight + chartMarginBottom);

    // Move charttooltip with mouse events
    d3.select(this.id).on('mousemove click', (e) => {
      let mouse = d3.pointer(e);
      updateChartTooltip(mouse[0], e.pageY);
    });

  }

  drawFuelChart(svg, pastData, forecastData, pastXCount, forecastXCount, pastXValues, forecastXValues) {
    // calculate cumulative values to make the stacked chart
    const fuels = ['Green_In', 'Wind', 'Hydro', 'Solar', 'Rooftop', 'Fossil_In', 'Gas', 'Coal'];
    let pastAccumulator = new Array(pastXCount).fill(0);
    this.pastFuelsStacked = fuels.map(fuel => {
      pastAccumulator = pastAccumulator.map((e,i) => e + pastData[fuel][i]);
      return pastAccumulator;
    });
    let forecastAccumulator = new Array(forecastXCount).fill(0);
    this.forecastFuelsStacked = fuels.map(fuel => {
      forecastAccumulator = forecastAccumulator.map((e,i) => e + forecastData[fuel][i]);
      return forecastAccumulator;
    });

    // thanks to the above ordering, we've also calculate greenness for free. TODO: should probably come from server.
    // this.pastGreenness = this.pastFuelsStacked[4]; // 4 happens to be the boundary between renewable and fossil in the ordering of columns

    svg.selectAll('.chart-area')
      .data(this.pastFuelsStacked.reverse())
      .enter()
      .append('path')
      .attr('class', 'chart-area')
      .attr('class', (d, i) => 'chart-area-' + fuels[fuels.length - 1 - i])
      .attr('d', d3.area()
        .x((d,i) => this.x(pastXValues[i]))
        .y0(this.y(0))
        .y1(d => this.y(d))
        );

    svg.selectAll('.chart-area')
      .data(this.forecastFuelsStacked.reverse())
      .enter()
      .append('path')
      .attr('class', 'chart-area')
      .attr('class', (d, i) => 'chart-area-' + fuels[fuels.length - 1 - i])
      .attr('d', d3.area()
        .x((d,i) => this.x(forecastXValues[i]))
        .y0(this.y(0))
        .y1(d => this.y(d))
        );

    // Draw the greenness plot line - past data
    svg.append('path')
      .datum(this.pastFuelsStacked[3])  // forecastFuelsStacked[3] is the last green fuel in the cumulative stack.
      .attr('class', 'chart-line')
      .style("stroke-dasharray", ("6, 3"))
      .attr('d', d3.line()
        .x((_,i) => this.x(pastXValues[i]))
        .y(d => this.y(d))
        );
    console.log(this.pastFuelsStacked[3]);

    // Draw the greenness plot line - forecast data
    svg.append('path')
      .datum(this.forecastFuelsStacked[3])  // forecastFuelsStacked[3] is the last green fuel in the cumulative stack.
      .attr('class', 'chart-line')
      .attr('d', d3.line()
        .x((_,i) => this.x(forecastXValues[i]))
        .y(d => this.y(d))
        );
  }
}

let greennessChart = new Chart('#greenness-chart');
let fuelChart = new Chart('#fuel-chart');
let priceChart = new Chart('#price-chart');


function resetChartTooltipToNow() {
  const nowXPos = greennessChart.x(greennessChart.now);
  const pageY = document.getElementById('greenness-chart').getBoundingClientRect().top + window.scrollY + 30;
  updateChartTooltip(nowXPos, pageY);
}

function updateChartTooltip(xPosPixels, pageY ) {
  // xPosPixels is the x coordinate (in pixels) within the chart svg's div, taken from the left edge of chart div (which by default is hidden off to the left)
  // 0 for xPosPixels === 6 days before today on the chart === 0 on the *range* of x
  // pageY is the target y coordinate within the *page* (window), ie as reported from mouse event. 

  const leftEdge = document.getElementById('greenness-chart').getBoundingClientRect().x;
  const pageX = leftEdge + xPosPixels;


  // move the line to mouse X
  d3.selectAll('.hover-line')
    .attr('x1', Math.round(xPosPixels - 1))
    .attr('x2', Math.round(xPosPixels - 1))
    .style('visibility', 'visible');

  // move tooltip box to near mouse and update it
  d3.selectAll('.chart-tooltip')
    .style('left', Math.round(pageX + 5) + 'px')
    .style('top', Math.round(pageY - 30) + 'px')
    .style('visibility', 'visible')
    .html(formatChartTooltip(greennessChart.x.invert(xPosPixels)));
}

function formatChartTooltip(time) {
  // formatChartTooltip() renders text for the chart tooltip

  let index = Math.round((time - greennessChart.baseTimeLocal ) / ONE_HOUR);

  if (index >= 2) {
    // forecast data
    index = index - 2; // first datapoint is actually 2 hrs after baseTimeLocal.
    index = Math.min(167, index);
    index = Math.max(0, index);

    let greenness = Math.round(latestForecasts.forecasts[getRegion() + '_Greenness'][index]);
    let price = Math.round(latestForecasts.forecasts[getRegion() + '_Price'][index]);

    return `${greenness}% Green
    <br>${Math.round(price/10)}¢/kWh
    <br>${formatHoursAmpm(time)} ${d3.timeFormat('%a')(time)}`;    

  } else {
    // past data - show legend

    // if mouse is over base_time or the gap between base_time and the first forecast, set index to base_time
    index = Math.min(0, index);

    // re-index into the past data    
    index = latestForecasts.pastTimestampsUtc.length - 1 + index; // last entry of pastGreenness is at baseTimeLocal
    index = Math.max(0, index);

    let greenness = Math.round(latestForecasts.past[getRegion() + '_Greenness'][index]);
    let price = Math.round(latestForecasts.past[getRegion() + '_Price'][index]);

    return `${greenness}% Green
    <br>${Math.round(price/10)}¢/kWh
    <br>${formatHoursAmpm(time)} ${d3.timeFormat('%a')(time)}`;
  }
}

// region selection buttons - restore the previously selected state from localStorage
let region = localStorage.getItem('region');
if (region === null) region = 'NSW1' // default to NSW
document.querySelector('input[type=radio][value=' + region + ']').checked = true;

// region selection buttons - redraw table when a region is selected
let radios = document.querySelectorAll('input[type=radio][name="region"]');
radios.forEach(radio => radio.addEventListener('change', render));

// left/right Buttons to move days when not showing entire chart
// updateChartScroll() moves the charts and table left/right
function updateChartScroll(newDay) {
  if (isMobile) {
    newDay = Math.max(0, newDay);
    newDay = Math.min(10, newDay); // 10 = total days (13) - days shown at once (3)
    document.getElementById('day-columns').style.left = -100.0 / 3 * newDay + '%';
  } else {
    newDay = Math.max(0, newDay);
    newDay = Math.min(6, newDay); // 6 = total days (13) - days shown at once (7)
    document.getElementById('day-columns').style.left = -100.0 / 7 * newDay + '%';
  }
  setTimeout(resetChartTooltipToNow, 210); //transition takes 200ms
  return newDay;
}
// when not showing the entire chart, currentDay keeps track of what day we're currently looking at. 0-indexed
let currentDay = 6; // by default, we're on 'today'

document.querySelectorAll('.left-button').forEach(e => {
  e.addEventListener('click', () => {
    currentDay = updateChartScroll(currentDay - 2);
  }
)});
document.querySelectorAll('.right-button').forEach(e => {
  e.addEventListener('click', () => {
    currentDay = updateChartScroll(currentDay + 2);
  }
)});

// TODO: react to changes
const isMobile = window.matchMedia("(max-width: 800px)").matches

// On mouseout from charts area, revert charttooltip to 'today'
d3.select('#day-columns').on('mouseleave', resetChartTooltipToNow);

// Text tooltips - toggle visibility on click
d3.selectAll('.has-tooltip').on('click', function(e) {
  e.stopPropagation(); // prevent click event bound to body from immediately closing this tooltip
  let tooltip = d3.select(this).select('.tooltip');
  let toggled = tooltip.style('visibility') === 'visible' ? 'hidden' : 'visible';
  tooltip.style('visibility', toggled);
});

// text tooltips - close on click
d3.select('body').on('click', () => {d3.selectAll('.tooltip').style('visibility', 'hidden');});

// collapsible areas in Q&A
d3.selectAll('.collapsible-heading').on('click', function() {
  this.classList.toggle('open-collapsible');
  let content = this.nextElementSibling;
  // slide open / closed.  But overflow: hidden is necessary and that hurts tooltips. 
  // content.style.maxHeight = content.style.maxHeight ? null : content.scrollHeight + 'px';
  content.style.display = content.style.display === 'block' ? 'none' : 'block';
})

// Get the data, store in global and draw the charts
fetch('latest_forecasts.json')
  .then(res => res.json())
  .then(forecasts => {
    // update global
    latestForecasts = forecasts;

    // Draw the table for the first time
    render();
  })
  .catch(err => { throw err });