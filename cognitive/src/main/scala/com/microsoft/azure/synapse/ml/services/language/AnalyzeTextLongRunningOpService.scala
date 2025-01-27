package com.microsoft.azure.synapse.ml.services.language

import com.microsoft.azure.synapse.ml.logging.{ FeatureNames, SynapseMLLogging }
import com.microsoft.azure.synapse.ml.services._
import com.microsoft.azure.synapse.ml.services.text.TextAnalyticsAutoBatch
import com.microsoft.azure.synapse.ml.services.vision.BasicAsyncReply
import com.microsoft.azure.synapse.ml.stages.{ FixedMiniBatchTransformer, FlattenBatch, HasBatchSize, UDFTransformer }
import org.apache.http.entity.{ AbstractHttpEntity, StringEntity }
import org.apache.spark.injections.UDFUtils
import org.apache.spark.ml.{ ComplexParamsReadable, NamespaceInjections, PipelineModel }
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ ArrayType, DataType, StructType }
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction

import java.net.URI

object AnalyzeTextLongRunningOpService extends ComplexParamsReadable[AnalyzeTextLongRunningOpService] with Serializable

class AnalyzeTextLongRunningOpService(override val uid: String) extends CognitiveServicesBaseNoHandler(uid)
                                                                        with BasicAsyncReply
                                                                        with HasCognitiveServiceInput
                                                                        with HasInternalJsonOutputParser
                                                                        with HasSetLocation
                                                                        with HasAPIVersion
                                                                        with HasCountryHint
                                                                        with TextAnalyticsAutoBatch
                                                                        with HasBatchSize
                                                                        with AnalyzeTextTaskParameters
                                                                        with SynapseMLLogging
                                                                        with AnalyzeTextServiceBaseParameters
                                                                        with HandleSentimentAnalysis {

  logClass(FeatureNames.AiServices.Language)

  def this() = this(Identifiable.randomUID("AnalyzeText"))

  setDefault(
    apiVersion -> Left("2022-05-01"),
    modelVersion -> Left("latest"),
    loggingOptOut -> Left(false),
    stringIndexType -> Left("TextElements_v8"),
    domain -> Left("none"),
    piiCategories -> Left(Seq.empty),
    opinionMining -> Left(false),
    batchSize -> 100, // scalastyle:ignore magic.number
    showStats -> Left(false),
    initialPollingDelay -> 1000, // scalastyle:ignore magic.number
    pollingDelay -> 1000, // scalastyle:ignore magic.number
    )

  override def urlPath: String = "/language/analyze-text/jobs"

  override protected def shouldSkip(row: Row): Boolean = if (emptyParamData(row, text)) {
    true
  } else {
    super.shouldSkip(row)
  }


  override protected def prepareEntity: Row => Option[AbstractHttpEntity] = row => {
    val body: String = getTypedKind match {
      case AnalysisTaskKind.SentimentAnalysis => createSentimentAnalysisLRORequest(row,
                                                                                   getValue(row, text),
                                                                                   getValueOpt(row, language),
                                                                                   getValue(row, modelVersion),
                                                                                   getValue(row, stringIndexType),
                                                                                   getValue(row, loggingOptOut))
      case _ => {
        "Hello World"
      }
    }

    Some(new StringEntity(body, "UTF-8"))
  }


  protected def postprocessResponse(responseOpt: Row): Option[Seq[Row]] = {
    Option(responseOpt).map { response =>
      val tasks = response.getAs[Row]("tasks")
      Seq(tasks.getAs[Seq[Row]]("items").head.getAs[Row]("results"))
    }
  }

  protected def postprocessResponseUdf: UserDefinedFunction = {
    val responseType = responseDataType.asInstanceOf[StructType]
    val outputType = responseType("tasks").dataType
                                          .asInstanceOf[StructType]("items").dataType
                                          .asInstanceOf[ArrayType].elementType.asInstanceOf[StructType]("results")
                                          .dataType
    UDFUtils.oldUdf(postprocessResponse _, outputType)
  }

  override protected def getInternalTransformer(schema: StructType): PipelineModel = {

    val batcher = if (shouldAutoBatch(schema)) {
      Some(new FixedMiniBatchTransformer().setBatchSize(getBatchSize))
    } else {
      None
    }
    val newSchema = batcher.map(_.transformSchema(schema)).getOrElse(schema)

    val pipe = super.getInternalTransformer(newSchema)

    val postprocess = new UDFTransformer()
      .setInputCol(getOutputCol)
      .setOutputCol(getOutputCol)
      .setUDF(postprocessResponseUdf)

    val flatten = if (shouldAutoBatch(schema)) {
      Some(new FlattenBatch())
    } else {
      None
    }

    NamespaceInjections.pipelineModel(
      Array(batcher, Some(pipe), Some(postprocess), flatten).flatten
      )
  }

  override protected def modifyPollingURI(originalURI: URI): URI = {
    if (getShowStats) {
      new URI(originalURI.toString + "&showStats=true")
    } else {
      originalURI
    }
  }

  override protected def responseDataType: DataType = getKind match {
    case "EntityRecognition" => EntityRecognitionResponse.schema
    case "LanguageDetection" => LanguageDetectionResponse.schema
    case "EntityLinking" => EntityLinkingResponse.schema
    case "KeyPhraseExtraction" => KeyPhraseExtractionResponse.schema
    case "PiiEntityRecognition" => PIIResponse.schema
    case "SentimentAnalysis" => SentimentAnlysisJobState.schema
  }
}